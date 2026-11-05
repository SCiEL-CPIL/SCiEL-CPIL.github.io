"""Root finding for ill-conditioned problems - improved SIQI with high-precision polish."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def find_root(f, x0):
    """
    Robust root finder for ill-conditioned problems.
    Improved SIQI with adaptive step control and high-precision final polish.
    """
    x0 = float(x0)
    deadline = time.monotonic() + 135.0
    max_evals = 495
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
        except (ValueError, OverflowError, ZeroDivisionError, TypeError, ArithmeticError):
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

    # Improved SIQI with better step control
    def siqi_root(x_start, fx_start, max_iter=60):
        """
        Safeguarded Inverse Quadratic Interpolation with adaptive step sizes.
        Uses quadratic interpolation when it's likely to converge, bisection otherwise.
        """
        if check_time():
            return None
        
        # Initialize bracket
        h0 = 1e-3 * max(1.0, abs(x_start))
        a = x_start - h0
        b = x_start + h0
        
        fa = eval_f(a)
        fb = eval_f(b)
        
        # Try multiple bracket sizes if first fails
        if fa is None or fb is None:
            for mult in [1e-4, 1e-2, 1e-1, 0.5, 1.0, 2.0, 5.0]:
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
        
        # Ensure sign change
        if fa * fb > 0:
            # Expand to find bracket
            step = max(1.0, abs(x_start)) * 0.5
            for _ in range(25):
                if check_time():
                    return None
                
                a_new = a - step
                fa_new = eval_f(a_new)
                if fa_new is not None:
                    if fa_new * fb < 0:
                        a, fa = a_new, fa_new
                        break
                    a, fa = a_new, fa_new
                
                b_new = b + step
                fb_new = eval_f(b_new)
                if fb_new is not None:
                    if fa * fb_new < 0:
                        b, fb = b_new, fb_new
                        break
                    b, fb = b_new, fb_new
                
                step *= 2
            else:
                return None
        
        # Now we have bracket [a, b] with fa*fb < 0
        # Choose initial points: use the one with smaller |f| as "best"
        if abs(fa) < abs(fb):
            x1, f1 = a, fa
            x2, f2 = b, fb
        else:
            x1, f1 = b, fb
            x2, f2 = a, fa
        
        # Third point for quadratic interpolation (midpoint)
        xm = (x1 + x2) / 2
        fm = eval_f(xm)
        if fm is None:
            return None
        
        # Ensure bracket still valid
        if fm * f1 < 0:
            x2, f2 = xm, fm
            a_br, b_br = a, xm
            fa_br, fb_br = fa, fm
        elif fm * f2 < 0:
            x1, f1 = xm, fm
            a_br, b_br = xm, b
            fa_br, fb_br = fm, fb
        else:
            # Shouldn't happen if fa*fb < 0
            return None
        
        # Track best point
        x_best, f_best = x1, f1
        if abs(f2) < abs(f_best):
            x_best, f_best = x2, f2
        
        # Main iteration loop
        for i in range(max_iter):
            if check_time():
                return None
            
            if abs(f_best) <= 1e-10:
                return x_best
            
            # Check for convergence
            if abs(b_br - a_br) < 1e-14 * max(1.0, abs((a_br + b_br) / 2)):
                break
            
            # Try inverse quadratic interpolation
            # We have three points: (x1, f1), (x2, f2), (x3, f3)
            # Use the two most recent and one endpoint
            x3, f3 = (x1, f1) if x2 == b_br else (x2, f2)
            if x3 == x1 and x2 == b_br:
                x3, f3 = a_br, fa_br
            elif x3 == x2 and x1 == b_br:
                x3, f3 = a_br, fa_br
            
            # Inverse quadratic interpolation formula
            xi = None
            try:
                if f1 != f2 and f1 != f3 and f2 != f3:
                    # Lagrange interpolation on (x, 1/f)
                    xi = (x1 * f2 * f3 / ((f1 - f2) * (f1 - f3)) +
                          x2 * f1 * f3 / ((f2 - f1) * (f2 - f3)) +
                          x3 * f1 * f2 / ((f3 - f1) * (f3 - f2)))
                else:
                    # Fall back to secant
                    if f2 != f1:
                        xi = x2 - f2 * (x2 - x1) / (f2 - f1)
            except (ZeroDivisionError, OverflowError):
                xi = None
            
            # Safeguard: if interpolation goes outside bracket, use bisection
            if xi is None or not math.isfinite(xi) or xi <= a_br or xi >= b_br:
                xi = (a_br + b_br) / 2
            else:
                # Additional safeguard: if interpolation is too close to an endpoint, use bisection
                if abs(xi - a_br) < 0.1 * (b_br - a_br) or abs(xi - b_br) < 0.1 * (b_br - a_br):
                    xi = (a_br + b_br) / 2
            
            fxi = eval_f(xi)
            if fxi is None:
                break
            
            if abs(fxi) <= 1e-10:
                return xi
            
            # Update bracket
            if fa_br * fxi < 0:
                b_br, fb_br = xi, fxi
                x2, f2 = xi, fxi
            else:
                a_br, fa_br = xi, fxi
                x1, f1 = xi, fxi
            
            # Update best
            if abs(fxi) < abs(f_best):
                x_best, f_best = xi, fxi
        
        return x_best if abs(f_best) <= 1e-8 else None

    # Newton's method with adaptive damping
    def newton_root(x_start, fx_start, max_iter=40):
        x = x_start
        fx = fx_start
        for i in range(max_iter):
            if check_time():
                return None
            if abs(fx) <= 1e-10:
                return x
            
            # Central difference for derivative
            h = 1e-6 * max(1.0, abs(x))
            fx_p = eval_f(x + h)
            fx_m = eval_f(x - h)
            
            if fx_p is None or fx_m is None:
                # Try one-sided
                fx_p = eval_f(x + h)
                if fx_p is None:
                    break
                d = (fx_p - fx) / h
            else:
                d = (fx_p - fx_m) / (2 * h)
            
            if d == 0 or not math.isfinite(d):
                break
            
            step = fx / d
            
            # Adaptive damping
            max_step = 0.5 * max(1.0, abs(x))
            if abs(step) > max_step:
                step = math.copysign(max_step, step)
            
            x_new = x - step
            fx_new = eval_f(x_new)
            if fx_new is None:
                break
            
            if abs(fx_new) < abs(fx):
                x, fx = x_new, fx_new
            else:
                # Try half step
                step *= 0.5
                x_new = x - step
                fx_new = eval_f(x_new)
                if fx_new is None:
                    break
                if abs(fx_new) < abs(fx):
                    x, fx = x_new, fx_new
                else:
                    break
        
        return x if abs(fx) <= 1e-8 else None

    # Secant method
    def secant_root(x1, fx1, x2, fx2, max_iter=30):
        for i in range(max_iter):
            if check_time():
                return None
            if abs(fx1) <= 1e-10:
                return x1
            if abs(fx2) <= 1e-10:
                return x2
            
            if fx1 == fx2:
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

    # Bisection
    def bisection_root(a, b, fa, fb, max_iter=80):
        for _ in range(max_iter):
            if check_time():
                return None
            mid = (a + b) / 2
            fm = eval_f(mid)
            if fm is None:
                break
            if abs(fm) <= 1e-10:
                return mid
            if fa * fm < 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
            if abs(b - a) < 1e-15 * max(1.0, abs(mid)):
                break
        mid = (a + b) / 2
        return mid

    # Find bracket
    def find_bracket(x_center, fx_center, max_expansions=30):
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

    # Main algorithm: try SIQI first, then Newton, then secant, then bisection
    result = siqi_root(x, fx)
    if result is not None:
        return result

    result = newton_root(x, fx)
    if result is not None:
        return result

    # Secant with perturbed point
    h = 1e-4 * max(1.0, abs(x))
    x2 = x + h
    fx2 = eval_f(x2)
    if fx2 is not None:
        result = secant_root(x, fx, x2, fx2)
        if result is not None:
            return result

    # Bisection
    bracket = find_bracket(x, fx)
    if bracket is not None:
        a, b, fa, fb = bracket
        result = bisection_root(a, b, fa, fb)
        if result is not None:
            return result

    # Multiple restarts
    rng = np.random.RandomState(12345)
    for attempt in range(15):
        if check_time():
            break
        
        radius = 1e-2 * (5 ** attempt) * max(1.0, abs(x0))
        x_try = x0 + rng.normal(0, radius)
        fx_try = eval_f(x_try)
        if fx_try is None:
            continue
        
        if abs(fx_try) < abs(best_fx):
            best_x, best_fx = x_try, fx_try
        
        if abs(fx_try) <= 1e-8:
            return x_try
        
        result = siqi_root(x_try, fx_try, max_iter=30)
        if result is not None:
            return result
        
        result = newton_root(x_try, fx_try, max_iter=20)
        if result is not None:
            return result

    # High-precision final polish
    if best_abs <= 1e-4 and best_fx is not None:
        # Try Newton with tighter tolerance
        result = newton_root(best_x, best_fx, max_iter=50)
        if result is not None:
            return result
        
        # Try secant polish
        h = 1e-7 * max(1.0, abs(best_x))
        x_p = best_x + h
        fx_p = eval_f(x_p)
        if fx_p is not None:
            result = secant_root(best_x, best_fx, x_p, fx_p, max_iter=40)
            if result is not None:
                return result
        
        # Try SIQI polish
        result = siqi_root(best_x, best_fx, max_iter=50)
        if result is not None:
            return result

    return best_x
# EVOLVE-BLOCK-END
