"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 for smooth polynomial equations.
    Uses a hybrid approach:
      1. Try Newton's method with finite-difference derivative for fast convergence.
      2. If Newton stalls or diverges, switch to secant method.
      3. If secant also struggles, use bisection on a bracket found by expansion.
    Includes time guard and call budget.
    """
    x0 = float(x0)
    tol = 1e-8
    start_time = time.time()
    time_limit = 140.0  # safety margin before 150s kill
    max_calls = 480
    call_count = 0
    
    def safe_f(x):
        nonlocal call_count
        if call_count >= max_calls:
            return None
        if time.time() - start_time > time_limit:
            return None
        call_count += 1
        try:
            val = f(x)
            if not math.isfinite(val):
                return None
            return val
        except:
            return None
    
    def time_ok():
        return time.time() - start_time < time_limit - 2.0
    
    def find_bracket(a, b, fa, fb):
        """Try to find a bracket [a, b] where f(a)*f(b) < 0."""
        if fa is None or fb is None:
            return None
        if fa * fb <= 0:
            return (a, b, fa, fb)
        
        # Expand from the point with smaller |f|
        if abs(fa) < abs(fb):
            a, b, fa, fb = b, a, fb, fa
        
        # Try expanding in both directions from b
        step = max(1.0, abs(b)) * 0.5
        for _ in range(50):
            if not time_ok():
                return None
            new_b = b + step
            new_fb = safe_f(new_b)
            if new_fb is None:
                break
            if fa * new_fb <= 0:
                return (a, new_b, fa, new_fb)
            step *= 1.5
        
        # Try expanding in the other direction
        step = max(1.0, abs(a)) * 0.5
        for _ in range(50):
            if not time_ok():
                return None
            new_a = a - step
            new_fa = safe_f(new_a)
            if new_fa is None:
                break
            if new_fa * fb <= 0:
                return (new_a, b, new_fa, fb)
            step *= 1.5
        
        return None
    
    # Initial evaluation
    f0 = safe_f(x0)
    if f0 is None:
        return x0
    if abs(f0) <= tol:
        return x0
    
    # Phase 1: Newton's method
    x = x0
    fx = f0
    best_x = x
    best_fx = fx
    
    for i in range(30):
        if not time_ok():
            break
        if abs(fx) <= tol:
            return x
        
        # Finite difference derivative
        h = 1e-7 * max(1.0, abs(x))
        fx_plus = safe_f(x + h)
        if fx_plus is None:
            break
        d = (fx_plus - fx) / h
        if d == 0 or not math.isfinite(d):
            break
        
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
        
        # Damping: if step is too large, halve it
        if abs(x_new - x) > 10 * max(1.0, abs(x)):
            x_new = x + 0.5 * (x_new - x)
        
        fx_new = safe_f(x_new)
        if fx_new is None:
            break
        
        if abs(fx_new) < abs(fx):
            x, fx = x_new, fx_new
            if abs(fx) < abs(best_fx):
                best_x, best_fx = x, fx
        else:
            # Newton is diverging; switch to secant
            break
    
    if abs(fx) <= tol:
        return x
    if abs(best_fx) <= tol:
        return best_x
    
    # Phase 2: Secant method
    # Use the last two points from Newton or start fresh
    x1, f1 = x, fx
    h = 1e-6 * max(1.0, abs(x))
    x2 = x + h
    f2 = safe_f(x2)
    if f2 is None:
        x2 = x - h
        f2 = safe_f(x2)
        if f2 is None:
            # Fall back to bisection
            pass
        else:
            pass
    
    if f2 is not None:
        for i in range(40):
            if not time_ok():
                break
            if abs(f1) <= tol:
                return x1
            if abs(f2) <= tol:
                return x2
            
            denom = f2 - f1
            if denom == 0 or not math.isfinite(denom):
                break
            
            x_new = x1 - f1 * (x2 - x1) / denom
            if not math.isfinite(x_new):
                break
            
            f_new = safe_f(x_new)
            if f_new is None:
                break
            
            if abs(f_new) < abs(f1):
                if abs(f_new) < abs(best_fx):
                    best_x, best_fx = x_new, f_new
                x1, f1 = x2, f2
                x2, f2 = x_new, f_new
            else:
                x1, f1 = x2, f2
                x2, f2 = x_new, f_new
            
            if abs(f_new) <= tol:
                return x_new
    
    if abs(best_fx) <= tol:
        return best_x
    
    # Phase 3: Bisection with bracket finding
    # Try to find a bracket using the best points we have
    bracket = find_bracket(best_x, x, best_fx, fx)
    if bracket is None:
        # Try expanding from x0
        bracket = find_bracket(x0, x0 + 1.0, safe_f(x0), safe_f(x0 + 1.0))
    if bracket is None:
        bracket = find_bracket(x0 - 1.0, x0, safe_f(x0 - 1.0), safe_f(x0))
    
    if bracket is not None:
        a, b, fa, fb = bracket
        for i in range(60):
            if not time_ok():
                break
            mid = (a + b) / 2
            fm = safe_f(mid)
            if fm is None:
                break
            if abs(fm) <= tol:
                return mid
            if abs(fm) < abs(best_fx):
                best_x, best_fx = mid, fm
            if fa * fm <= 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
            if abs(b - a) < 1e-14:
                break
    
    if abs(best_fx) <= tol:
        return best_x
    
    # Return the best point found
    return best_x
# EVOLVE-BLOCK-END
