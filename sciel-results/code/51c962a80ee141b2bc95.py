"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import brentq, minimize_scalar

def find_root(f, x0):
    """
    Robust root finder for ill-conditioned problems.
    Strategy:
      1. Cheap bracket search (sign change -> brentq; else minimize |f| via golden-section)
      2. Safeguarded inverse quadratic interpolation (SIQI) with bisection fallback
      3. Newton's method with adaptive damping
      4. Secant method
      5. Multiple restarts if initial guess is poor
      6. Deadline guard to stay within time limits
    """
    x0 = float(x0)
    deadline = time.monotonic() + 135.0  # safety margin before 150s kill
    max_evals = 490
    eval_count = [0]
    
    best_x = x0
    best_fx = None
    best_abs = math.inf

    def eval_f(x):
        nonlocal best_x, best_fx, best_abs
        if time.monotonic() > deadline:
            return None
        if eval_count[0] >= max_evals:
            return None
        try:
            val = f(x)
            eval_count[0] += 1
            if not math.isfinite(val):
                return None
            if abs(val) < best_abs:
                best_x = x
                best_fx = val
                best_abs = abs(val)
            return val
        except (ValueError, OverflowError, ZeroDivisionError, TypeError):
            eval_count[0] += 1
            return None

    def check_time():
        return time.monotonic() > deadline

    x = x0
    fx = eval_f(x)
    if fx is None:
        return best_x
    if abs(fx) <= 1e-8:
        return x

    # --- Safeguarded Inverse Quadratic Interpolation (SIQI) ---
    def try_siqi(x_start, fx_start, max_iter=50):
        """
        SIQI: Uses quadratic interpolation in inverse space with safeguarding.
        Maintains a bracket [a, b] and uses interpolation to pick the next point,
        falling back to bisection if interpolation goes outside the bracket.
        """
        if check_time():
            return None
        
        # Initialize bracket around x_start
        h0 = 1e-4 * max(1.0, abs(x_start))
        a = x_start - h0
        b = x_start + h0
        
        fa = eval_f(a)
        fb = eval_f(b)
        
        if fa is None or fb is None:
            for mult in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
                if check_time():
                    return None
                h = mult * max(1.0, abs(x_start))
                a = x_start - h
                b = x_start + h
                fa = eval_f(a)
                fb = eval_f(b)
                if fa is not None and fb is not None:
                    break
            if fa is None or fb is None:
                return None
        
        # Ensure we have a sign change
        if fa * fb > 0:
            step = max(1.0, abs(x_start)) * 0.5
            for _ in range(20):
                if check_time():
                    return None
                a_new = a - step
                fa_new = eval_f(a_new)
                if fa_new is not None and fa_new * fb < 0:
                    a, fa = a_new, fa_new
                    break
                if fa_new is not None:
                    a, fa = a_new, fa_new
                
                b_new = b + step
                fb_new = eval_f(b_new)
                if fb_new is not None and fa * fb_new < 0:
                    b, fb = b_new, fb_new
                    break
                if fb_new is not None:
                    b, fb = b_new, fb_new
                
                step *= 2
            else:
                return None
        
        # Now we have a bracket [a, b] with fa*fb < 0
        if abs(fa) < abs(fb):
            x1, f1 = a, fa
            x2, f2 = b, fb
        else:
            x1, f1 = b, fb
            x2, f2 = a, fa
        
        # Evaluate midpoint to establish three points
        xm = (x1 + x2) / 2
        fm = eval_f(xm)
        if fm is None:
            return None
        
        if fm * f1 < 0:
            x2, f2 = xm, fm
        elif fm * f2 < 0:
            x1, f1 = xm, fm
        else:
            return None
        
        # Set bracket
        if fa * fb < 0:
            a_br, b_br = a, b
            fa_br, fb_br = fa, fb
        elif fm * fa < 0:
            a_br, b_br = a, xm
            fa_br, fb_br = fa, fm
        else:
            a_br, b_br = xm, b
            fa_br, fb_br = fm, fb
        
        # Keep track of best point
        if abs(fa_br) < abs(fb_br):
            x_best_br, f_best_br = a_br, fa_br
        else:
            x_best_br, f_best_br = b_br, fb_br
        
        # Third point for interpolation
        if x_best_br == a_br:
            x3, f3 = b_br, fb_br
        else:
            x3, f3 = a_br, fa_br
        
        # Main SIQI loop
        for i in range(max_iter):
            if check_time():
                return None
            
            if abs(f_best_br) <= 1e-12:
                return x_best_br
            
            # Try inverse quadratic interpolation
            # Fit quadratic through (x1,f1), (x2,f2), (x3,f3) in inverse space
            # i.e., fit x as a quadratic function of 1/f
            # For simplicity, use the three points we have
            x_a, f_a = a_br, fa_br
            x_b, f_b = b_br, fb_br
            
            # Use the best point and the other bracket endpoint
            if abs(f_best_br) < abs(f_a) and abs(f_best_br) < abs(f_b):
                # best is interior, use endpoints for IQI
                xa, fa_i = x_a, f_a
                xb, fb_i = x_b, f_b
                xc, fc_i = x_best_br, f_best_br
            else:
                # Use best and one endpoint
                if abs(f_a) <= abs(f_b):
                    xa, fa_i = x_best_br, f_best_br
                    xb, fb_i = x_a, f_a
                    xc, fc_i = x_b, f_b
                else:
                    xa, fa_i = x_best_br, f_best_br
                    xb, fb_i = x_b, f_b
                    xc, fc_i = x_a, f_a
            
            # IQI: solve for x where 1/f = 0
            # Using Lagrange interpolation in (1/f, x) space
            try:
                if fa_i == 0 or fb_i == 0 or fc_i == 0:
                    raise ZeroDivisionError
                inv_fa = 1.0 / fa_i
                inv_fb = 1.0 / fb_i
                inv_fc = 1.0 / fc_i
                
                # Lagrange polynomial: x = A * inv_f * (inv_f - inv_fb)*(inv_f - inv_fc) + ...
                # We want x at inv_f = 0
                denom_a = (inv_fa - inv_fb) * (inv_fa - inv_fc)
                denom_b = (inv_fb - inv_fa) * (inv_fb - inv_fc)
                denom_c = (inv_fc - inv_fa) * (inv_fc - inv_fb)
                
                if abs(denom_a) < 1e-30 or abs(denom_b) < 1e-30 or abs(denom_c) < 1e-30:
                    raise ZeroDivisionError
                
                x_iqi = (xa * inv_fb * inv_fc / denom_a + 
                         xb * inv_fa * inv_fc / denom_b + 
                         xc * inv_fa * inv_fb / denom_c)
            except (ZeroDivisionError, OverflowError, FloatingPointError):
                x_iqi = None
            
            # Safeguard: if x_iqi is outside bracket, use bisection
            if x_iqi is None or not math.isfinite(x_iqi) or x_iqi <= a_br or x_iqi >= b_br:
                x_new = (a_br + b_br) / 2
            else:
                x_new = x_iqi
            
            f_new = eval_f(x_new)
            if f_new is None:
                break
            
            if abs(f_new) <= 1e-12:
                return x_new
            
            # Update bracket
            if f_new * fa_br < 0:
                b_br, fb_br = x_new, f_new
            else:
                a_br, fa_br = x_new, f_new
            
            # Update best
            if abs(f_new) < abs(f_best_br):
                x_best_br, f_best_br = x_new, f_new
            
            # Update third point
            if x_best_br == a_br:
                x3, f3 = b_br, fb_br
            else:
                x3, f3 = a_br, fa_br
            
            # Check convergence
            if abs(b_br - a_br) < 1e-15 * max(1.0, abs(x_new)):
                break
        
        return x_best_br if abs(f_best_br) <= 1e-8 else None

    # --- Newton's method with damping ---
    def try_newton(x_start, fx_start, max_iter=40):
        x = x_start
        fx = fx_start
        for i in range(max_iter):
            if check_time():
                return None
            if abs(fx) <= 1e-12:
                return x
            
            h = 1e-7 * max(1.0, abs(x))
            fx_plus = eval_f(x + h)
            if fx_plus is None:
                break
            d = (fx_plus - fx) / h
            if d == 0 or not math.isfinite(d):
                break
            
            step = fx / d
            # Damp the step
            max_step = 0.5 * max(abs(x), 1.0)
            if abs(step) > max_step:
                step = math.copysign(max_step, step)
            
            x_new = x - step
            fx_new = eval_f(x_new)
            if fx_new is None:
                break
            
            if abs(fx_new) < abs(fx):
                x, fx = x_new, fx_new
            else:
                # Try smaller step
                step *= 0.25
                x_new = x - step
                fx_new = eval_f(x_new)
                if fx_new is None:
                    break
                if abs(fx_new) < abs(fx):
                    x, fx = x_new, fx_new
                else:
                    break
        return x if abs(fx) <= 1e-8 else None

    # --- Secant method ---
    def try_secant(x1, fx1, x2, fx2, max_iter=40):
        for i in range(max_iter):
            if check_time():
                return None
            if abs(fx1) <= 1e-12:
                return x1
            if abs(fx2) <= 1e-12:
                return x2
            
            if abs(fx1 - fx2) < 1e-30:
                break
            
            x_new = x2 - fx2 * (x2 - x1) / (fx2 - fx1)
            fx_new = eval_f(x_new)
            if fx_new is None:
                break
            
            x1, fx1 = x2, fx2
            x2, fx2 = x_new, fx_new
        
        best = x1 if abs(fx1) <= abs(fx2) else x2
        best_fx = fx1 if abs(fx1) <= abs(fx2) else fx2
        return best if abs(best_fx) <= 1e-8 else None

    # --- Bracket finding ---
    def find_bracket(x_center, fx_center, max_expansions=25):
        a, b = x_center, x_center
        fa, fb = fx_center, fx_center
        
        step_size = max(1.0, abs(x_center)) * 0.1
        for _ in range(max_expansions):
            if check_time():
                return None
            
            a_new = a - step_size
            fa_new = eval_f(a_new)
            if fa_new is not None:
                if fa_new * fb < 0:
                    return (a_new, b, fa_new, fb)
                a, fa = a_new, fa_new
            
            b_new = b + step_size
            fb_new = eval_f(b_new)
            if fb_new is not None:
                if fa * fb_new < 0:
                    return (a, b_new, fa, fb_new)
                b, fb = b_new, fb_new
            
            step_size *= 2
        
        return None

    # --- Bisection ---
    def try_bisection(a, b, fa, fb, max_iter=60):
        for _ in range(max_iter):
            if check_time():
                return None
            mid = (a + b) / 2
            fm = eval_f(mid)
            if fm is None:
                break
            if abs(fm) <= 1e-12:
                return mid
            if fa * fm < 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
            if abs(b - a) < 1e-15 * max(1.0, abs(mid)):
                break
        mid = (a + b) / 2
        return mid

    # --- Golden section on |f|^2 ---
    def try_minimize(fx_at_x0, x_start):
        """Minimize |f(x)|^2 using golden section search."""
        if check_time():
            return None
        
        # Find an interval where |f| has a minimum
        h = 1e-3 * max(1.0, abs(x_start))
        a = x_start - h
        b = x_start + h
        
        fa = eval_f(a)
        fb = eval_f(b)
        
        if fa is None or fb is None:
            for mult in [1e-2, 1e-1, 1.0, 10.0]:
                if check_time():
                    return None
                h = mult * max(1.0, abs(x_start))
                a = x_start - h
                b = x_start + h
                fa = eval_f(a)
                fb = eval_f(b)
                if fa is not None and fb is not None:
                    break
            if fa is None or fb is None:
                return None
        
        # Expand if needed to find a valley
        for _ in range(15):
            if check_time():
                return None
            if abs(fx_at_x0) < min(abs(fa), abs(fb)):
                # x0 is already in a valley
                break
            h = (b - a) / 2
            a_new = a - h
            b_new = b + h
            fa_new = eval_f(a_new)
            fb_new = eval_f(b_new)
            if fa_new is not None and fb_new is not None:
                a, b = a_new, b_new
                fa, fb = fa_new, fb_new
            else:
                break
        
        # Golden section search
        gr = (math.sqrt(5) - 1) / 2  # 0.618...
        
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        fc = eval_f(c)
        fd = eval_f(d)
        
        if fc is None or fd is None:
            return None
        
        for _ in range(40):
            if check_time():
                break
            
            if abs(fc) < abs(fd):
                b = d
                fb = fd
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = eval_f(c)
            else:
                a = c
                fa = fc
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = eval_f(d)
            
            if fc is None or fd is None:
                break
            
            if abs(b - a) < 1e-14 * max(1.0, abs((a+b)/2)):
                break
        
        # Best of the last evaluations
        candidates = [(c, fc), (d, fd), (a, fa), (b, fb)]
        best_c = min(candidates, key=lambda t: abs(t[1]) if t[1] is not None else math.inf)
        if best_c[1] is not None and abs(best_c[1]) <= 1e-8:
            return best_c[0]
        return None

    # === Main search sequence ===
    
    # 1. Try brentq if we can find a bracket
    bracket = find_bracket(x, fx, max_expansions=15)
    if bracket is not None:
        a_br, b_br, fa_br, fb_br = bracket
        try:
            # Use scipy's brentq with our eval_f to count evaluations
            root = brentq(lambda xx: eval_f(xx) if eval_f(xx) is not None else 1e10, 
                         a_br, b_br, xtol=1e-14, rtol=1e-14, maxiter=100)
            if root is not None and abs(f(root)) <= 1e-8:
                return root
        except (ValueError, RuntimeError, OverflowError, ZeroDivisionError):
            pass
        # Fall through to SIQI if brentq failed
    
    # 2. Try SIQI
    result = try_siqi(x, fx)
    if result is not None:
        return result
    
    # 3. Try Newton
    result = try_newton(x, fx)
    if result is not None:
        return result
    
    # 4. Try secant with a perturbed point
    h = 1e-5 * max(1.0, abs(x))
    x2 = x + h
    fx2 = eval_f(x2)
    if fx2 is not None:
        result = try_secant(x, fx, x2, fx2)
        if result is not None:
            return result
    
    # 5. Try golden section on |f|^2
    result = try_minimize(fx, x)
    if result is not None:
        return result
    
    # 6. Try bisection with a found bracket
    bracket = find_bracket(x, fx)
    if bracket is not None:
        a, b, fa, fb = bracket
        result = try_bisection(a, b, fa, fb)
        if result is not None:
            return result
    
    # 7. Multiple restarts with increasing radius
    rng = np.random.RandomState(12345)
    for attempt in range(15):
        if check_time() or eval_count[0] >= max_evals - 5:
            break
        
        radius = 1e-3 * (5 ** attempt) * max(1.0, abs(x0))
        x_try = x0 + rng.normal(0, radius)
        fx_try = eval_f(x_try)
        if fx_try is None:
            continue
        
        if abs(fx_try) < abs(best_fx):
            best_x, best_fx = x_try, fx_try
        
        if abs(fx_try) <= 1e-8:
            return x_try
        
        # Try SIQI from this point
        result = try_siqi(x_try, fx_try, max_iter=25)
        if result is not None:
            return result
        
        # Try Newton from this point
        result = try_newton(x_try, fx_try, max_iter=20)
        if result is not None:
            return result
        
        # Try secant
        h2 = 1e-6 * max(1.0, abs(x_try))
        x_try2 = x_try + h2
        fx_try2 = eval_f(x_try2)
        if fx_try2 is not None:
            result = try_secant(x_try, fx_try, x_try2, fx_try2, max_iter=20)
            if result is not None:
                return result
    
    # 8. Final polish: if we have a good candidate, try to refine further
    if best_abs <= 1e-4 and best_fx is not None:
        result = try_newton(best_x, best_fx, max_iter=30)
        if result is not None:
            return result
        
        h = 1e-8 * max(1.0, abs(best_x))
        x_p = best_x + h
        fx_p = eval_f(x_p)
        if fx_p is not None:
            result = try_secant(best_x, best_fx, x_p, fx_p, max_iter=30)
            if result is not None:
                return result
        
        # Try a very tight SIQI around best_x
        h = 1e-6 * max(1.0, abs(best_x))
        a_p = best_x - h
        b_p = best_x + h
        fa_p = eval_f(a_p)
        fb_p = eval_f(b_p)
        if fa_p is not None and fb_p is not None and fa_p * fb_p < 0:
            result = try_bisection(a_p, b_p, fa_p, fb_p, max_iter=60)
            if result is not None:
                return result
    
    return best_x
# EVOLVE-BLOCK-END
