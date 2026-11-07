import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 115.0  # Safety margin for 120s limit
    
    calls = 0
    best_x = x0
    best_f_abs = float('inf')
    cache = {}
    
    def _f(x):
        nonlocal calls, best_x, best_f_abs
        if time.time() - start_time > TIME_LIMIT:
            raise TimeoutError("Time limit exceeded")
        
        # Check cache
        if x in cache:
            return cache[x]
            
        try:
            val = float(f(x))
        except Exception:
            cache[x] = float('nan')
            return float('nan')
            
        if not math.isfinite(val):
            cache[x] = float('nan')
            return float('nan')
            
        calls += 1
        cache[x] = val
        
        abs_val = abs(val)
        if abs_val < best_f_abs:
            best_f_abs = abs_val
            best_x = x
            
        return val

    x0 = float(x0)
    fx0 = _f(x0)
    
    if abs(fx0) <= 1e-8:
        return x0

    # Strategy: Global Sign-Change Sweep with Adaptive Refinement
    # 1. Generate a set of points covering a wide range, prioritizing local area.
    # 2. Sort them by x-value to detect sign changes between neighbors.
    # 3. Evaluate sequentially until a bracket is found or budget is exhausted.
    # 4. Use Illinois algorithm (robust bisection + secant) to refine the root.
    
    candidates = set()
    
    # Local dense grid
    # If the root is close to x0, we want high resolution.
    for i in range(-100, 101):
        candidates.add(x0 + i * 0.01)
        
    # Expanding rings
    step = 0.1
    for _ in range(10):
        for i in range(-20, 21):
            candidates.add(x0 + i * step)
        step *= 2.0
        
    # Very wide sparse coverage
    for i in range(-200, 201):
        candidates.add(x0 + i * 100.0)
        
    # Extreme
    for i in range(-20, 21):
        candidates.add(x0 + i * 1000.0)
        
    # Add some random points to catch weird oscillations? No, deterministic is better for reproducibility.
    # But we can add a few "jumps" just in case.
    # Actually, the grid above is quite comprehensive.

    # Sort candidates to ensure we check neighbors in order for sign change detection
    # We prioritize points closer to x0 to find local roots faster, BUT
    # sign change detection requires x-order. 
    # Compromise: We evaluate in a specific order that covers local area first, 
    # but we store results in a dict to check for brackets later?
    # No, checking brackets later requires knowing which points are neighbors in x-space.
    # If we evaluate out of x-order, we might miss a bracket between two evaluated points
    # if a third point (not evaluated yet) lies between them and changes the sign.
    # However, if we evaluate ALL points in the set, we can then sort the evaluated points
    # and check for sign changes.
    
    # Let's do this:
    # 1. Create a list of unique x values.
    # 2. Sort them by distance to x0? No, sort by x-value.
    # 3. But we want to prioritize local. 
    #    Idea: Create multiple "shells" of points.
    #    Shell 0: [x0-0.1, x0+0.1]
    #    Shell 1: [x0-0.5, x0+0.5]
    #    ...
    #    Evaluate Shell 0 points (sorted by x). Check for brackets.
    #    If found, refine.
    #    If not, add Shell 1 points to the pool? Or just evaluate them?
    #    This is complex.
    
    # Simpler approach:
    # Just sort all candidates by x-value.
    # Iterate through them.
    # To prioritize local, we can "jump" to the local region first?
    # No, that breaks the neighbor property.
    
    # Alternative:
    # We can just evaluate a subset of points in x-order, but choose the subset to be dense locally.
    # The grid generation above already does this:
    # - 0.01 step for +/- 1
    # - 0.1 step for +/- 2
    # - 0.2 step for +/- 4
    # etc.
    
    # So, if we sort all these points by x, the local region will be densely packed.
    # The global region will be sparsely packed.
    # If we iterate through the sorted list, we will evaluate local points first (if we start from x0 and go out? No, sorted by x starts from min x).
    # To prioritize local, we should sort by distance to x0?
    # If we sort by distance to x0, we lose the x-neighbor property for bracket detection.
    
    # Let's try a hybrid:
    # 1. Generate candidates.
    # 2. Sort by x-value.
    # 3. Iterate.
    # 4. BUT, we can "fast-forward" to the region near x0?
    #    No, we need to check the whole range for a root.
    
    # Let's stick to the sorted-by-x approach.
    # The "priority" is handled by the density of the grid.
    # If the root is local, we will find it quickly because the grid is dense there.
    # If the root is global, we might waste calls on local points that don't have roots.
    # But local points are few (200 points for +/- 1).
    # Global points are sparse.
    
    # To improve:
    # We can check if f(x0) is close to 0. (Already done)
    # We can check a few points around x0 first.
    # Let's evaluate a small local window first: [x0-0.5, x0+0.5] with step 0.01.
    # If a bracket is found, refine.
    # If not, proceed to the global sweep.
    
    # Phase 1: Local Search
    local_candidates = sorted([x0 + i * 0.01 for i in range(-50, 51)])
    prev_x = None
    prev_f = None
    bracket = None
    
    for x in local_candidates:
        if calls >= 100:
            break
        if time.time() - start_time > TIME_LIMIT:
            break
            
        fx = _f(x)
        
        if abs(fx) <= 1e-8:
            return x
            
        if prev_x is not None and prev_f is not None:
            if (prev_f > 0 and fx < 0) or (prev_f < 0 and fx > 0):
                bracket = (prev_x, x)
                break
            # Also check if one is very close to 0 (handled by abs check above)
            
        prev_x = x
        prev_f = fx
        
    if bracket is not None:
        a, b = bracket
        fa = _f(a)
        fb = _f(b)
        
        # Refine with Illinois
        for _ in range(100):
            if calls >= 490:
                break
            if time.time() - start_time > TIME_LIMIT:
                break
                
            denom = fb - fa
            if abs(denom) < 1e-300:
                x_new = (a + b) / 2.0
            else:
                x_new = a - fa * (b - a) / denom
                
            if not (a < x_new < b):
                x_new = (a + b) / 2.0
                
            f_new = _f(x_new)
            
            if abs(f_new) <= 1e-8:
                return x_new
                
            if fa * f_new < 0:
                b = x_new
                fb = f_new
            else:
                a = x_new
                fa = f_new
                # Illinois tweak: if x_new is closer to b, halve fb
                if abs(x_new - b) < abs(b - a):
                    fb = fb / 2.0
                    
        # Return best found
        candidates_refine = [best_x, a, b]
        best_refine_x = best_x
        best_refine_val = best_f_abs
        for x in candidates_refine:
            if x in cache:
                val = abs(cache[x])
            else:
                val = abs(_f(x))
            if val < best_refine_val:
                best_refine_val = val
                best_refine_x = x
        return best_refine_x

    # Phase 2: Global Search
    # Generate global candidates
    global_candidates = set()
    step = 0.1
    for _ in range(8):
        for i in range(-20, 21):
            global_candidates.add(x0 + i * step)
        step *= 2.0
        
    for i in range(-100, 101):
        global_candidates.add(x0 + i * 100.0)
        
    for i in range(-10, 11):
        global_candidates.add(x0 + i * 1000.0)
        
    # Remove points already evaluated in local search?
    # We can just evaluate them, the cache will handle duplicates.
    
    unique_sorted = sorted(list(global_candidates))
    
    prev_x = None
    prev_f = None
    
    for x in unique_sorted:
        if calls >= 350:
            break
        if time.time() - start_time > TIME_LIMIT:
            break
            
        fx = _f(x)
        
        if abs(fx) <= 1e-8:
            return x
            
        if prev_x is not None and prev_f is not None:
            if (prev_f > 0 and fx < 0) or (prev_f < 0 and fx > 0):
                bracket = (prev_x, x)
                break
            
        prev_x = x
        prev_f = fx
        
    if bracket is None:
        return best_x
        
    a, b = bracket
    fa = _f(a)
    fb = _f(b)
    
    # Refine with Illinois
    for _ in range(100):
        if calls >= 490:
            break
        if time.time() - start_time > TIME_LIMIT:
            break
            
        denom = fb - fa
        if abs(denom) < 1e-300:
            x_new = (a + b) / 2.0
        else:
            x_new = a - fa * (b - a) / denom
            
        if not (a < x_new < b):
            x_new = (a + b) / 2.0
            
        f_new = _f(x_new)
        
        if abs(f_new) <= 1e-8:
            return x_new
            
        if fa * f_new < 0:
            b = x_new
            fb = f_new
        else:
            a = x_new
            fa = f_new
            if abs(x_new - b) < abs(b - a):
                fb = fb / 2.0
                
    candidates_refine = [best_x, a, b]
    best_refine_x = best_x
    best_refine_val = best_f_abs
    for x in candidates_refine:
        if x in cache:
            val = abs(cache[x])
        else:
            val = abs(_f(x))
        if val < best_refine_val:
            best_refine_val = val
            best_refine_x = x
    return best_refine_x

# EVOLVE-BLOCK-END
