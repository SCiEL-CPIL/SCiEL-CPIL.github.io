import math
import time
import scipy.optimize as opt

def find_root(f, x0):
    START_TIME = time.time()
    TIME_LIMIT = 110.0  # Safety margin before 120s/150s limit
    MAX_CALLS = 495
    
    calls = 0
    cache = {}
    best_x = x0
    best_f = float('inf')
    found = False

    def safe_f(x):
        nonlocal calls, best_x, best_f, found
        if found:
            return 0.0
        if calls >= MAX_CALLS:
            return float('nan')
        if time.time() - START_TIME > TIME_LIMIT:
            return float('nan')
        
        # Use exact float key for cache
        if x in cache:
            val = cache[x]
        else:
            calls += 1
            try:
                val = f(x)
                val = float(val)
            except Exception:
                val = float('nan')
            cache[x] = val
            
        if math.isfinite(val):
            abs_val = abs(val)
            if abs_val < best_f:
                best_f = abs_val
                best_x = x
                if best_f <= 1e-8:
                    found = True
        return val

    # Evaluate starting point
    fx0 = safe_f(x0)
    if found:
        return best_x

    # Strategy: Coarse-to-fine scanning
    # Start with larger steps to find a bracket quickly with fewer calls.
    # If no bracket is found, decrease step size.
    # We scan outwards from x0 in both directions.
    
    steps = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]
    
    for step in steps:
        if found:
            return best_x
        
        # Scan positive direction
        prev_x = x0
        prev_fx = fx0
        # Limit range to avoid too many calls in one pass
        # 100 steps of 1.0 is 100 calls. 100 steps of 0.01 is 100 calls.
        # We want to cover a reasonable range.
        max_steps = 100
        
        for i in range(1, max_steps + 1):
            x_new = x0 + i * step
            fx_new = safe_f(x_new)
            if found:
                return best_x
            
            if math.isfinite(prev_fx) and math.isfinite(fx_new):
                # Check for sign change
                if prev_fx * fx_new < 0:
                    bracket = (prev_x, x_new)
                    result = refine_root(bracket, safe_f, cache)
                    if result is not None:
                        return result
                # Check for near-zero (in case root is touched but not crossed, though unlikely for oscillatory)
                # safe_f already updates best_x/best_f and found flag
                
            prev_x = x_new
            prev_fx = fx_new
            
            if calls > MAX_CALLS - 20:
                break

        if found:
            return best_x
        if calls > MAX_CALLS - 20:
            break

        # Scan negative direction
        prev_x = x0
        prev_fx = fx0
        
        for i in range(1, max_steps + 1):
            x_new = x0 - i * step
            fx_new = safe_f(x_new)
            if found:
                return best_x
            
            if math.isfinite(prev_fx) and math.isfinite(fx_new):
                if prev_fx * fx_new < 0:
                    bracket = (x_new, prev_x)
                    result = refine_root(bracket, safe_f, cache)
                    if result is not None:
                        return result
                
            prev_x = x_new
            prev_fx = fx_new
            
            if calls > MAX_CALLS - 20:
                break

        if found:
            return best_x
        if calls > MAX_CALLS - 20:
            break

    # If no bracket found via sign change, return best point found
    return best_x

def refine_root(bracket, safe_f, cache):
    a, b = bracket
    
    # Ensure a < b
    if a > b:
        a, b = b, a
    
    fa = cache.get(a, safe_f(a))
    fb = cache.get(b, safe_f(b))
    
    if not math.isfinite(fa) or not math.isfinite(fb):
        return None
    
    # Check if endpoints are already good enough
    if abs(fa) <= 1e-8:
        return a
    if abs(fb) <= 1e-8:
        return b

    # Try brentq first for fast convergence
    try:
        root = opt.brentq(safe_f, a, b, xtol=1e-14, rtol=1e-14, maxiter=50)
        if math.isfinite(root):
            fr = safe_f(root)
            if math.isfinite(fr) and abs(fr) <= 1e-8:
                return root
    except Exception:
        pass
    
    # Fallback: Secant Method with Bisection safeguard
    # Secant is faster than bisection for smooth functions.
    x0_sec = a
    x1_sec = b
    f0 = fa
    f1 = fb
    
    # Ensure f0 and f1 have opposite signs
    # If not, we might have a problem, but brentq should have handled it.
    # If brentq failed, maybe the function is not continuous?
    # For now, stick to bisection fallback if secant fails or diverges.
    
    for _ in range(50):
        if abs(f1) < abs(f0):
            x0_sec, x1_sec = x1_sec, x0_sec
            f0, f1 = f1, f0
            
        # Secant step
        if f1 == f0:
            # Degenerate case, use bisection
            x_new = (x0_sec + x1_sec) / 2.0
        else:
            x_new = x1_sec - f1 * (x1_sec - x0_sec) / (f1 - f0)
            
        # Safeguard: keep x_new within [x0_sec, x1_sec]
        if not (min(x0_sec, x1_sec) < x_new < max(x0_sec, x1_sec)):
            x_new = (x0_sec + x1_sec) / 2.0
            
        f_new = safe_f(x_new)
        
        if not math.isfinite(f_new):
            break
            
        if abs(f_new) <= 1e-8:
            return x_new
            
        # Update bracket
        if f0 * f_new < 0:
            x1_sec = x_new
            f1 = f_new
        else:
            x0_sec = x_new
            f0 = f_new

    # Return best point
    candidates = [(a, abs(fa)), (b, abs(fb)), (x0_sec, abs(f0)), (x1_sec, abs(f1))]
    best = min(candidates, key=lambda x: x[1])
    if best[1] <= 1e-8:
        return best[0]
        
    return None
