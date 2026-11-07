import math
import time

def find_root(f, x0):
    TOL = 1e-8
    MAX_CALLS = 490
    deadline = time.time() + 145.0
    
    calls = 0
    best_x = float(x0)
    best_abs = float('inf')
    
    def safe_f(x):
        nonlocal calls, best_x, best_abs
        if calls >= MAX_CALLS or time.time() > deadline:
            return float('inf')
        calls += 1
        try:
            val = f(x)
            if not math.isfinite(val):
                return float('inf')
            a = abs(val)
            if a < best_abs:
                best_abs = a
                best_x = x
            return val
        except Exception:
            return float('inf')
    
    x0 = float(x0)
    
    # First, evaluate at x0
    fx0 = safe_f(x0)
    if abs(fx0) <= TOL:
        return x0
    if not math.isfinite(fx0):
        # x0 is in a bad region, try nearby
        for dx in [0.1, -0.1, 0.5, -0.5, 1.0, -1.0]:
            x_try = x0 + dx
            fx_try = safe_f(x_try)
            if abs(fx_try) <= TOL:
                return x_try
            if math.isfinite(fx_try):
                best_x = x_try
                best_abs = abs(fx_try)
                break
        return best_x
    
    # Try to find a bracket near x0 using small steps
    # For oscillatory functions, the period is likely O(1) or O(10)
    # Try multiple scales of local search
    
    bracket_a = None
    bracket_b = None
    
    # Try progressively larger local windows
    for scale in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]:
        if calls >= MAX_CALLS - 50 or time.time() > deadline:
            break
        
        # Check left and right of x0
        xl = x0 - scale
        xr = x0 + scale
        
        fxl = safe_f(xl)
        fxr = safe_f(xr)
        
        if not math.isfinite(fxl) or not math.isfinite(fxr):
            continue
        
        # Check if sign change between xl and x0
        if fx0 * fxl < 0:
            bracket_a = xl
            bracket_b = x0
            break
        
        # Check if sign change between x0 and xr
        if fx0 * fxr < 0:
            bracket_a = x0
            bracket_b = xr
            break
        
        # Check if sign change between xl and xr (x0 might be outside)
        if fxl * fxr < 0:
            bracket_a = xl
            bracket_b = xr
            break
    
    if bracket_a is not None:
        # Use bisection + IQUO (inverse quadratic interpolation) hybrid to refine
        a, b = bracket_a, bracket_b
        fa = safe_f(a)
        fb = safe_f(b)
        
        # Ensure fa and fb have opposite signs
        if fa * fb > 0:
            # Re-evaluate to be safe
            fa = safe_f(a)
            fb = safe_f(b)
        
        # Track previous point for IQUO
        x_prev = None
        f_prev = None
        
        for _ in range(60):
            if calls >= MAX_CALLS - 10 or time.time() > deadline:
                break
            if fa * fb > 0:
                break
            if abs(b - a) < 1e-15:
                break
            
            # Try IQUO if we have 3 points
            xn = None
            if x_prev is not None and f_prev is not None:
                # Use the three most recent points: (a, fa), (b, fb), (x_prev, f_prev)
                # IQUO formula for finding root
                denom1 = (fa - f_prev)
                denom2 = (fb - f_prev)
                denom3 = (fa - fb)
                
                if abs(denom1) > 1e-15 and abs(denom2) > 1e-15 and abs(denom3) > 1e-15:
                    # Inverse quadratic interpolation
                    num = fa * fb * (x_prev - b) / (fa - f_prev) / (fa - fb) \
                        + fa * f_prev * (b - a) / (f_prev - fb) / (fa - f_prev) \
                        + fb * f_prev * (a - x_prev) / (fb - fa) / (fb - f_prev)
                    
                    # Actually, let's use the standard IQUO formula
                    # x_new = (a*fa*fb/(fa-fb)/(fa-f_prev) + b*fb*f_prev/(fb-fa)/(fb-f_prev) + x_prev*f_prev*fa/(f_prev-fb)/(f_prev-fa))
                    # Simpler: use secant as fallback, try IQUO
                    try:
                        t1 = a * fa * fb / (fa - fb) / (fa - f_prev)
                        t2 = b * fb * f_prev / (fb - fa) / (fb - f_prev)
                        t3 = x_prev * f_prev * fa / (f_prev - fb) / (f_prev - fa)
                        xn = t1 + t2 + t3
                    except (ZeroDivisionError, OverflowError):
                        xn = None
            
            # If IQUO failed or not available, use secant/Newton
            if xn is None:
                if fb != fa and abs(b - a) > 1e-15:
                    deriv = (fb - fa) / (b - a)
                else:
                    deriv = 1.0
                
                if abs(deriv) > 1e-15:
                    xn = b - fb / deriv
                else:
                    xn = (a + b) / 2.0
            
            # Keep xn in bracket
            if not (a < xn < b):
                xn = (a + b) / 2.0
            
            fxn = safe_f(xn)
            
            if abs(fxn) <= TOL:
                return xn
            
            if not math.isfinite(fxn):
                xn = (a + b) / 2.0
                fxn = safe_f(xn)
                if not math.isfinite(fxn):
                    break
            
            # Update bracket
            if fa * fxn < 0:
                # New point is in [a, xn]
                x_prev = b
                f_prev = fb
                b = xn
                fb = fxn
            else:
                # New point is in [xn, b]
                x_prev = a
                f_prev = fa
                a = xn
                fa = fxn
        
        # Return the better endpoint
        if abs(fa) < abs(fb):
            return a
        else:
            return b
    
    # No bracket found - use secant method iteratively
    # This works well for smooth oscillatory functions
    x_prev = x0 - 0.1
    fx_prev = safe_f(x_prev)
    x_curr = x0
    fx_curr = safe_f(x_curr)
    
    if abs(fx_curr) <= TOL:
        return x_curr
    
    for _ in range(80):
        if calls >= MAX_CALLS - 5 or time.time() > deadline:
            break
        
        if abs(fx_prev) < abs(fx_curr):
            # Swap to keep convergence
            x_prev, x_curr = x_curr, x_prev
            fx_prev, fx_curr = fx_curr, fx_prev
        
        if abs(fx_curr) <= TOL:
            return x_curr
        
        if fx_prev == fx_curr:
            break
        
        x_new = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
        
        # Damp if step is too large
        if abs(x_new - x_curr) > 10.0 * max(1.0, abs(x_curr)):
            x_new = x_curr - 10.0 * max(1.0, abs(x_curr)) * (1.0 if fx_curr > 0 else -1.0)
        
        fx_new = safe_f(x_new)
        
        if abs(fx_new) <= TOL:
            return x_new
        
        if not math.isfinite(fx_new):
            # Try smaller step
            x_new = x_curr - 0.1 * (x_curr - x_prev)
            fx_new = safe_f(x_new)
            if not math.isfinite(fx_new):
                break
        
        x_prev, fx_prev = x_curr, fx_curr
        x_curr, fx_curr = x_new, fx_new
    
    # Final high-precision polish: if best_abs is small but > TOL, try to refine
    if best_abs < 1e-4 and math.isfinite(best_abs):
        # Do a few secant steps starting from best_x with a nearby point
        xp = best_x + 0.01
        fxp = safe_f(xp)
        xq = best_x
        fxq = safe_f(xq)
        
        for _ in range(20):
            if calls >= MAX_CALLS - 2 or time.time() > deadline:
                break
            if abs(fxq) <= TOL:
                return xq
            if fxp == fxq:
                break
            xnew = xq - fxq * (xq - xp) / (fxq - fxp)
            fxnew = safe_f(xnew)
            if abs(fxnew) <= TOL:
                return xnew
            if not math.isfinite(fxnew):
                break
            xp, fxp = xq, fxq
            xq, fxq = xnew, fxnew
    
    # Return best found
    return best_x
