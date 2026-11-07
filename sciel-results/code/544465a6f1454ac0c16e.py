import math
import time

def find_root(f, x0):
    TOL = 1e-8
    TOL_POLISH = 1e-12
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
    for scale in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]:
        if calls >= MAX_CALLS - 30 or time.time() > deadline:
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
        
        # Track previous points for IQUO
        x_c = (a + b) / 2.0
        fc = safe_f(x_c)
        
        for _ in range(60):
            if calls >= MAX_CALLS - 10 or time.time() > deadline:
                break
            if fa * fb > 0:
                break
            if abs(b - a) < 1e-15:
                break
            
            # Try IQUO first, fall back to bisection
            xn = None
            if abs(fb) > 1e-300 and abs(fc) > 1e-300 and abs(fa) > 1e-300:
                # Inverse quadratic interpolation
                denom_a = (fb - fc) * (fa - fc)
                denom_b = (fa - fb) * (fa - fc)
                denom_c = (fa - fb) * (fb - fc)
                if abs(denom_a) > 1e-300 and abs(denom_b) > 1e-300 and abs(denom_c) > 1e-300:
                    num = (a * denom_a + b * denom_b + x_c * denom_c)
                    den = denom_a + denom_b + denom_c
                    if abs(den) > 1e-300:
                        xn = num / den
            
            # If IQUO failed or is outside bracket, use bisection
            if xn is None or not (a < xn < b):
                xn = (a + b) / 2.0
            
            fxn = safe_f(xn)
            
            if abs(fxn) <= TOL:
                # Do a final polish pass
                return _polish(f, xn, safe_f, TOL_POLISH, MAX_CALLS, deadline, calls)
            
            if not math.isfinite(fxn):
                xn = (a + b) / 2.0
                fxn = safe_f(xn)
                if not math.isfinite(fxn):
                    break
            
            # Update bracket - keep the two points with opposite signs
            if fa * fxn < 0:
                b = xn
                fb = fxn
            else:
                a = xn
                fa = fxn
            
            # Update x_c for IQUO
            x_c = xn
            fc = fxn
        
        # Final polish on the best endpoint
        if abs(fa) < abs(fb):
            return _polish(f, a, safe_f, TOL_POLISH, MAX_CALLS, deadline, calls)
        else:
            return _polish(f, b, safe_f, TOL_POLISH, MAX_CALLS, deadline, calls)
    
    # No bracket found - use secant method iteratively
    # This works well for smooth oscillatory functions
    x_prev = x0 - 0.1
    fx_prev = safe_f(x_prev)
    x_curr = x0
    fx_curr = safe_f(x_curr)
    
    if abs(fx_curr) <= TOL:
        return _polish(f, x_curr, safe_f, TOL_POLISH, MAX_CALLS, deadline, calls)
    
    for _ in range(80):
        if calls >= MAX_CALLS - 5 or time.time() > deadline:
            break
        
        if abs(fx_prev) < abs(fx_curr):
            # Swap to keep convergence
            x_prev, x_curr = x_curr, x_prev
            fx_prev, fx_curr = fx_curr, fx_prev
        
        if abs(fx_curr) <= TOL:
            return _polish(f, x_curr, safe_f, TOL_POLISH, MAX_CALLS, deadline, calls)
        
        if fx_prev == fx_curr:
            break
        
        x_new = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
        
        # Damp if step is too large
        if abs(x_new - x_curr) > 10.0 * max(1.0, abs(x_curr)):
            x_new = x_curr - 10.0 * max(1.0, abs(x_curr)) * (1.0 if fx_curr > 0 else -1.0)
        
        fx_new = safe_f(x_new)
        
        if abs(fx_new) <= TOL:
            return _polish(f, x_new, safe_f, TOL_POLISH, MAX_CALLS, deadline, calls)
        
        if not math.isfinite(fx_new):
            # Try smaller step
            x_new = x_curr - 0.1 * (x_curr - x_prev)
            fx_new = safe_f(x_new)
            if not math.isfinite(fx_new):
                break
        
        x_prev, fx_prev = x_curr, fx_curr
        x_curr, fx_curr = x_new, fx_new
    
    # Return best found
    return best_x

def _polish(f, x_start, safe_f, tol_polish, MAX_CALLS, deadline, calls):
    """Final high-precision polish pass using secant method with tighter tolerance."""
    if calls >= MAX_CALLS - 3 or time.time() > deadline:
        return x_start
    
    x_curr = x_start
    fx_curr = safe_f(x_curr)
    if abs(fx_curr) <= tol_polish:
        return x_curr
    
    # Use a small perturbation for secant
    dx = 1e-8
    x_prev = x_curr - dx
    fx_prev = safe_f(x_prev)
    if not math.isfinite(fx_prev):
        return x_curr
    
    for _ in range(10):
        if calls >= MAX_CALLS or time.time() > deadline:
            break
        if abs(fx_curr) <= tol_polish:
            return x_curr
        if fx_prev == fx_curr:
            break
        
        x_new = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
        fx_new = safe_f(x_new)
        
        if not math.isfinite(fx_new):
            break
        
        if abs(fx_new) < abs(fx_curr):
            x_prev, fx_prev = x_curr, fx_curr
            x_curr, fx_curr = x_new, fx_new
        else:
            x_prev, fx_prev = x_new, fx_new
    
    # Return the point with smaller |f|
    if abs(fx_curr) <= abs(fx_prev):
        return x_curr
    else:
        return x_prev
