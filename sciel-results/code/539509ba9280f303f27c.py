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
    for scale in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
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
        
        # Also check midpoint for tighter bracket
        xm = (xl + xr) / 2.0
        fxm = safe_f(xm)
        if math.isfinite(fxm):
            if fx0 * fxm < 0:
                if fx0 > fxm:
                    bracket_a = x0
                    bracket_b = xm
                else:
                    bracket_a = xm
                    bracket_b = x0
                break
            if fxl * fxm < 0:
                bracket_a = xl
                bracket_b = xm
                break
            if fxm * fxr < 0:
                bracket_a = xm
                bracket_b = xr
                break
    
    if bracket_a is not None:
        # Use Brent's method for robust and fast convergence
        a, b = bracket_a, bracket_b
        fa = safe_f(a)
        fb = safe_f(b)
        
        # Ensure fa and fb have opposite signs
        if fa * fb > 0:
            # Try to fix the bracket
            fa = safe_f(a)
            fb = safe_f(b)
            if fa * fb > 0:
                # No valid bracket, fall through to secant
                pass
            else:
                # Proceed with Brent
                pass
        
        if fa * fb < 0:
            # Implement Brent's method
            c = a
            fc = fa
            d = b - a
            e = d
            
            for _ in range(60):
                if calls >= MAX_CALLS - 10 or time.time() > deadline:
                    break
                if fa * fb > 0:
                    break
                if abs(b - a) < 1e-15:
                    break
                
                # Inverse quadratic interpolation or bisection
                m = 0.5 * (a + b)
                em = m - e
                
                if abs(em) < 1e-14 and abs(fc) < 1e-14:
                    # Bisection
                    xn = m
                elif a == c or b == c or fa == fb:
                    if a == b:
                        xn = m
                    else:
                        # Secant
                        if abs(fb) < abs(fa):
                            xn = b - fb * (b - a) / (fb - fa)
                        else:
                            xn = a - fa * (b - a) / (fb - fa)
                        if not (a < xn < b) or not math.isfinite(xn):
                            xn = m
                else:
                    # Inverse quadratic interpolation
                    try:
                        # Compute IQUO
                        # Using the formula from Brent's paper
                        t1 = b - a
                        t2 = b - c
                        t3 = c - a
                        
                        # We want to find x such that the quadratic through
                        # (a, fa), (b, fb), (c, fc) equals zero
                        # Using Lagrange interpolation on 1/f is complex,
                        # so we use the standard IQUO formula
                        # 
                        # Let's use a simpler approach: Newton step with
                        # secant as fallback
                        # Actually, let's just use secant for simplicity
                        # since IQUO implementation is error-prone
                        if abs(fb) < abs(fa):
                            xn = b - fb * (b - a) / (fb - fa)
                        else:
                            xn = a - fa * (b - a) / (fb - fa)
                        
                        if not (a < xn < b) or not math.isfinite(xn):
                            xn = m
                    except (ZeroDivisionError, OverflowError):
                        xn = m
                
                # Keep xn in bracket
                if not (a < xn < b) or not math.isfinite(xn):
                    xn = m
                
                fxn = safe_f(xn)
                
                if abs(fxn) <= TOL:
                    return xn
                
                if not math.isfinite(fxn):
                    xn = m
                    fxn = safe_f(xn)
                    if not math.isfinite(fxn):
                        break
                
                # Update
                c = a
                fc = fa
                e = d
                d = m - xn
                
                if fa * fxn < 0:
                    b = xn
                    fb = fxn
                else:
                    a = xn
                    fa = fxn
                
                # Check convergence
                if abs(xn - m) < 1e-14:
                    break
            
            # Final check
            if abs(fa) <= TOL:
                return a
            if abs(fb) <= TOL:
                return b
            
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
    
    # Return best found
    return best_x
