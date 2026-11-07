"""
Root finder using global golden-section search over a coarse grid,
then local secant/Newton refinement for fast convergence.
"""
import time
import math

def find_root(f, x0):
    """
    Finds a root x of f(x) such that |f(x)| <= 1e-8.
    Uses global search to find a good starting point, then local refinement.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    TOL = 1e-9
    MAX_CALLS = 480
    
    x = float(x0)
    call_count = 0
    cache = {}
    
    def _elapsed():
        return time.time() - start_time
    
    def _time_ok():
        return _elapsed() < deadline and call_count < MAX_CALLS
    
    def _f(x_val):
        """Cached function evaluation with call counting."""
        nonlocal call_count
        if call_count >= MAX_CALLS:
            return None
        key = x_val
        if key in cache:
            return cache[key]
        try:
            val = f(key)
            call_count += 1
            if not math.isfinite(val):
                cache[key] = None
                return None
            cache[key] = val
            return val
        except Exception:
            call_count += 1
            return None

    # Phase 0: Check if x0 is already a root
    fx0 = _f(x)
    if fx0 is None:
        return x
    if abs(fx0) <= TOL:
        return x
    
    # Phase 1: Global coarse grid search to find the best point
    # We'll search over a wide range around x0
    # Use a logarithmic scale to cover both small and large ranges
    
    best_x = x
    best_abs = abs(fx0)
    
    # Define a set of candidate points to evaluate
    # Start with points near x0, then expand outward
    scale = max(1.0, abs(x))
    
    # Coarse grid: ~100 points total
    # We'll use a combination of local and global points
    candidates = []
    
    # Local points around x0
    for mult in [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
        candidates.append(x + mult * scale)
        candidates.append(x - mult * scale)
    
    # Also add some intermediate points
    for mult in [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]:
        candidates.append(x + mult * scale * 0.5)
        candidates.append(x - mult * scale * 0.5)
    
    # Evaluate all candidates, tracking the best
    for cx in candidates:
        if not _time_ok():
            break
        if not math.isfinite(cx):
            continue
        cf = _f(cx)
        if cf is None:
            continue
        if abs(cf) < best_abs:
            best_abs = abs(cf)
            best_x = cx
        if abs(cf) <= TOL:
            return cx
    
    # Phase 2: Local refinement using secant method
    # Start from best_x
    
    x_curr = best_x
    fx_curr = _f(x_curr)
    if fx_curr is None:
        return best_x
    
    if abs(fx_curr) <= TOL:
        return x_curr
    
    # Try secant method with a nearby point
    dx = 1e-4 * max(1.0, abs(x_curr))
    x_prev = x_curr + dx
    fx_prev = _f(x_prev)
    if fx_prev is None:
        # Try the other direction
        x_prev = x_curr - dx
        fx_prev = _f(x_prev)
        if fx_prev is None:
            return x_curr
    
    # Secant iteration
    for _ in range(50):
        if not _time_ok():
            break
        if abs(fx_curr) <= TOL:
            return x_curr
        if abs(fx_prev) <= TOL:
            return x_prev
        
        denom = fx_curr - fx_prev
        if abs(denom) < 1e-15:
            break
        
        x_new = x_curr - fx_curr * (x_curr - x_prev) / denom
        if not math.isfinite(x_new):
            break
        
        x_prev, fx_prev = x_curr, fx_curr
        x_curr = x_new
        fx_curr = _f(x_curr)
        if fx_curr is None:
            break
    
    # Phase 3: If secant didn't converge, try bisection with a bracket
    # Try to find a bracket around best_x
    if abs(fx_curr) > TOL:
        # Expand to find a bracket
        a = best_x - 1e-3 * max(1.0, abs(best_x))
        b = best_x + 1e-3 * max(1.0, abs(best_x))
        fa = _f(a)
        fb = _f(b)
        
        if fa is not None and fb is not None and fa * fb < 0:
            # Bisection
            for _ in range(60):
                if not _time_ok():
                    break
                if abs(fa) <= TOL:
                    return a
                if abs(fb) <= TOL:
                    return b
                
                mid = (a + b) / 2.0
                fmid = _f(mid)
                if fmid is None:
                    break
                if abs(fmid) <= TOL:
                    return mid
                
                if fa * fmid < 0:
                    b, fb = mid, fmid
                else:
                    a, fa = mid, fmid
            
            if abs(fa) < abs(fb):
                return a
            else:
                return b
    
    # Return the best estimate
    if abs(fx_curr) <= TOL:
        return x_curr
    return best_x
