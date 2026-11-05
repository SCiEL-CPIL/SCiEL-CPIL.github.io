"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def find_root(f, x0):
    """
    Robust root finder for ill-conditioned problems.
    Primary: Illinois algorithm (inverse bisection) for guaranteed bracket convergence.
    Secondary: Newton/secant polish for speed when derivative is reliable.
    Fallback: bisection and random restarts.
    """
    x0 = float(x0)
    deadline = time.monotonic() + 140.0  # safety margin before 150s kill
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

    # Illinois Algorithm: Inverse bisection with weight reduction
    def try_illinois(x_start, fx_start, max_iter=80):
        """
        Illinois method: A variant of the bisection method that uses
        inverse bisection (secant-like step) but reduces the weight of
        the less important endpoint when the new point has the same sign
        as the more important one. This guarantees convergence and is
        robust for ill-conditioned functions.
        """
        if check_time():
            return None
        
        # Step 1: Find a bracket [a, b] where f(a)*f(b) < 0
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
        
        # Expand if no sign change
        if fa * fb > 0:
            step = max(1.0, abs(x_start)) * 0.5
            found = False
            for _ in range(20):
                if check_time():
                    return None
                a_new = a - step
                fa_new = eval_f(a_new)
                if fa_new is not None:
                    if fa_new * fb < 0:
                        a, fa = a_new, fa_new
                        found = True
                        break
                    a, fa = a_new, fa_new
                
                b_new = b + step
                fb_new = eval_f(b_new)
                if fb_new is not None:
                    if fa * fb_new < 0:
                        b, fb = b_new, fb_new
                        found = True
                        break
                    b, fb = b_new, fb_new
                
                step *= 2
            if not found:
                return None
        
        # Now we have a bracket [a, b] with fa*fb < 0
        # Initialize weights
        wa = 1.0
        wb = 1.0
        
        # Main Illinois loop
        for i in range(max_iter):
            if check_time():
                break
            
            if abs(fa) <= 1e-12:
                return a
            if abs(fb) <= 1e-12:
                return b
            
            # Inverse bisection (weighted secant)
            # c = (wa*a*fb - wb*b*fa) / (wa*fb - wb*fa)
            denom = wa * fb - wb * fa
            if abs(denom) < 1e-300:
                # Fallback to simple midpoint
                c = (a + b) / 2
            else:
                c = (wa * a * fb - wb * b * fa) / denom
            
            # Safeguard: if c is outside [a,b] or too close to an endpoint, use midpoint
            if c <= a or c >= b or abs(c - a) < 1e-15 * max(1.0, abs(a)) or abs(c - b) < 1e-15 * max(1.0, abs(b)):
                c = (a + b) / 2
            
            fc = eval_f(c)
            if fc is None:
                break
            
            if abs(fc) <= 1e-12:
                return c
            
            # Update bracket and weights
            if fc * fa < 0:
                # Root is in [a, c]
                b, fb = c, fc
                # Reduce weight of b (the endpoint that is now further from root)
                wb *= 0.5
            else:
                # Root is in [c, b]
                a, fa = c, fc
                # Reduce weight of a
                wa *= 0.5
            
            # Check convergence
            if abs(b - a) < 1e-15 * max(1.0, abs(c)):
                break
        
        # Return the point with smallest |f|
        if abs(fa) < abs(fb):
            return a
        else:
            return b
    
    # Newton's method with damping
    def try_newton(x_start, fx_start, max_iter=30):
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
            # Damp the step if it's too large
            if abs(step) > abs(x) * 0.5:
                step = math.copysign(min(abs(step), abs(x) * 0.5), step)
            
            x_new = x - step
            fx_new = eval_f(x_new)
            if fx_new is None:
                break
            
            if abs(fx_new) < abs(fx):
                x, fx = x_new, fx_new
            else:
                # Try smaller step
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
    def try_secant(x1, fx1, x2, fx2, max_iter=30):
        for i in range(max_iter):
            if check_time():
                return None
            if abs(fx1) <= 1e-12:
                return x1
            if abs(fx2) <= 1e-12:
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
    
    # Try Illinois first (most robust for ill-conditioned)
    result = try_illinois(x, fx)
    if result is not None and abs(eval_f(result) if eval_f(result) is not None else math.inf) <= 1e-8:
        return result
    if result is not None:
        fx_res = eval_f(result)
        if fx_res is not None and abs(fx_res) < abs(best_abs):
            best_x = result
            best_abs = abs(fx_res)
    
    # Try Newton
    result = try_newton(x, fx)
    if result is not None:
        return result
    
    # Try secant with a perturbed point
    h = 1e-5 * max(1.0, abs(x))
    x2 = x + h
    fx2 = eval_f(x2)
    if fx2 is not None:
        result = try_secant(x, fx, x2, fx2)
        if result is not None:
            return result
    
    # Multiple restarts with Illinois
    rng = np.random.RandomState(42)
    for attempt in range(10):
        if check_time():
            break
        
        radius = 1e-3 * (10 ** attempt) * max(1.0, abs(x0))
        x_try = x0 + rng.normal(0, radius)
        fx_try = eval_f(x_try)
        if fx_try is None:
            continue
        
        if abs(fx_try) < abs(best_abs):
            best_x = x_try
            best_abs = abs(fx_try)
        
        if abs(fx_try) <= 1e-8:
            return x_try
        
        result = try_illinois(x_try, fx_try, max_iter=30)
        if result is not None:
            fx_res = eval_f(result)
            if fx_res is not None and abs(fx_res) <= 1e-8:
                return result
            if fx_res is not None and abs(fx_res) < abs(best_abs):
                best_x = result
                best_abs = abs(fx_res)
        
        result = try_newton(x_try, fx_try, max_iter=15)
        if result is not None:
            return result
    
    return best_x
# EVOLVE-BLOCK-END
