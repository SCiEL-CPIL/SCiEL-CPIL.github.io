import math
import time
import scipy.optimize as opt

def find_root(f, x0):
    START_TIME = time.time()
    TIME_LIMIT = 110.0  # Safety margin before 120s hard limit
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

    # Evaluate initial point
    fx0 = safe_f(x0)
    if found:
        return best_x

    # Helper to check if two points form a bracket
    def is_bracket(xa, xb, fa, fb):
        if not math.isfinite(fa) or not math.isfinite(fb):
            return False
        return fa * fb < 0

    # Refine a bracket to high precision
    def refine_root(a, b, fa, fb):
        nonlocal best_x, best_f, found
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
        
        # Try brentq first for fast convergence
        try:
            root = opt.brentq(safe_f, a, b, xtol=1e-12, rtol=1e-12, maxiter=30)
            if math.isfinite(root):
                fr = safe_f(root)
                if math.isfinite(fr) and abs(fr) <= 1e-8:
                    return root
        except Exception:
            pass

        # Fallback: Bisection
        # We need to ensure we have valid fa, fb for the current a, b
        # If brentq consumed calls, we might need to re-evaluate or use cached
        # But safe_f handles caching. Let's just do a few bisection steps to be safe.
        
        # Re-fetch or compute fa, fb if they are stale or missing (though they should be in cache)
        if not math.isfinite(fa):
            fa = safe_f(a)
        if not math.isfinite(fb):
            fb = safe_f(b)

        for _ in range(40):
            if found:
                return best_x
            mid = (a + b) / 2.0
            fm = safe_f(mid)
            if not math.isfinite(fm):
                break
            
            if abs(fm) <= 1e-8:
                return mid

            if fa * fm <= 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm
        
        # Return the best candidate from the final interval
        candidates = [a, b]
        if 'mid' in locals():
            candidates.append(mid)
        
        # Pick the one with the smallest |f|
        best_local = min(candidates, key=lambda x: abs(cache.get(x, safe_f(x))))
        return best_local

    # Strategy:
    # 1. Try a very fine local scan around x0 to catch nearby roots.
    # 2. If not found, expand the search range with increasing step sizes.
    # 3. Use a "frontier" approach to avoid re-scanning.
    
    # Phase 1: Fine local scan
    # Scan +/- 1.0 with step 0.01 (200 points total)
    h = 0.01
    max_range = 1.0
    
    prev_x = x0
    prev_fx = fx0
    
    # Positive direction
    for i in range(1, int(max_range/h) + 1):
        x_new = x0 + i * h
        fx_new = safe_f(x_new)
        if found:
            return best_x
            
        if is_bracket(prev_x, x_new, prev_fx, fx_new):
            return refine_root(prev_x, x_new, prev_fx, fx_new)
            
        prev_x = x_new
        prev_fx = fx_new
        
    # Negative direction
    prev_x = x0
    prev_fx = fx0
    for i in range(1, int(max_range/h) + 1):
        x_new = x0 - i * h
        fx_new = safe_f(x_new)
        if found:
            return best_x
            
        if is_bracket(x_new, prev_x, fx_new, prev_fx):
            return refine_root(x_new, prev_x, fx_new, prev_fx)
            
        prev_x = x_new
        prev_fx = fx_new

    # Phase 2: Coarser wider scans
    # If we didn't find it in +/- 1.0, try wider ranges.
    # We maintain the frontier to continue scanning from where we left off.
    
    pos_frontier = x0 + max_range
    neg_frontier = x0 - max_range
    pos_fx = cache.get(pos_frontier, safe_f(pos_frontier))
    neg_fx = cache.get(neg_frontier, safe_f(neg_frontier))
    
    if found:
        return best_x

    configs = [
        (0.05, 5.0),   # Step 0.05, Range 5.0
        (0.2, 20.0),   # Step 0.2, Range 20.0
        (1.0, 100.0),  # Step 1.0, Range 100.0
        (5.0, 500.0),  # Step 5.0, Range 500.0
        (20.0, 2000.0) # Step 20.0, Range 2000.0
    ]
    
    for step, target_range in configs:
        # Positive direction
        x_curr = pos_frontier
        fx_curr = pos_fx
        # Calculate how many steps to reach target_range
        # Note: x_curr is already at max_range from previous phase or start
        # We want to go from x_curr to x0 + target_range
        
        # Ensure we don't step backwards
        if x_curr < x0 + max_range:
             x_curr = x0 + max_range
             fx_curr = cache.get(x_curr, safe_f(x_curr))
        
        while x_curr < x0 + target_range:
            if found:
                return best_x
            x_new = x_curr + step
            if x_new > x0 + target_range:
                break
            fx_new = safe_f(x_new)
            if is_bracket(x_curr, x_new, fx_curr, fx_new):
                return refine_root(x_curr, x_new, fx_curr, fx_new)
            x_curr = x_new
            fx_curr = fx_new
            
        pos_frontier = x_curr
        pos_fx = fx_curr
        
        # Negative direction
        x_curr = neg_frontier
        fx_curr = neg_fx
        if x_curr > x0 - max_range:
             x_curr = x0 - max_range
             fx_curr = cache.get(x_curr, safe_f(x_curr))
        
        while x_curr > x0 - target_range:
            if found:
                return best_x
            x_new = x_curr - step
            if x_new < x0 - target_range:
                break
            fx_new = safe_f(x_new)
            if is_bracket(x_new, x_curr, fx_new, fx_curr):
                return refine_root(x_new, x_curr, fx_new, fx_curr)
            x_curr = x_new
            fx_curr = fx_new
            
        neg_frontier = x_curr
        neg_fx = fx_curr

    # If no bracket found, return best_x
    return best_x
