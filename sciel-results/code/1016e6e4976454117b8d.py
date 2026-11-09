"""Root finder for 3 fixed oscillatory equations with call-budget awareness."""
import math
import time
import numpy as np

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8, starting near x0.
    Uses a hybrid approach:
      1. Evaluate at x0; if good enough, return immediately.
      2. Use a dense uniform grid with caching to find sign-change brackets.
      3. Apply high-precision bisection within the identified bracket.
      4. Track best (smallest |f|) throughout; return best if deadline approaches.
    """
    deadline = time.time() + 140.0  # safety margin before 150s kill
    tol = 1e-10  # tighter than required 1e-8

    x = float(x0)
    
    # Track best
    best_x = x
    cache = {}  # x -> f(x)
    
    def cached_f(x_val):
        """Evaluate f with caching."""
        x_key = float(x_val)
        if x_key in cache:
            return cache[x_key]
        try:
            result = f(x_key)
            if result is None or not math.isfinite(result):
                result = float('inf')
            cache[x_key] = result
            return result
        except Exception:
            cache[x_key] = float('inf')
            return float('inf')
    
    best_f = cached_f(x)
    best_abs = abs(best_f) if math.isfinite(best_f) else float('inf')
    
    if best_abs <= tol:
        return best_x

    # Phase 1: Dense uniform grid scan to find sign-change brackets
    # Start with a moderate range around x0
    base_range = max(1.0, abs(x))
    
    # Try different grid scales to find a bracket
    for scale in [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
        if time.time() > deadline - 5.0:
            break
        
        half_width = scale * base_range
        n_points = 100  # 100 points in the grid
        
        # Create uniform grid from x - half_width to x + half_width
        grid_x = np.linspace(x - half_width, x + half_width, n_points)
        grid_f = np.array([cached_f(xi) for xi in grid_x])
        
        # Update best
        finite_mask = np.isfinite(grid_f)
        if np.any(finite_mask):
            min_idx = np.argmin(np.abs(grid_f[finite_mask]))
            actual_idx = np.where(finite_mask)[0][min_idx]
            if abs(grid_f[actual_idx]) < best_abs:
                best_abs = abs(grid_f[actual_idx])
                best_x = grid_x[actual_idx]
                best_f = grid_f[actual_idx]
            if best_abs <= tol:
                return best_x
        
        # Find sign changes
        for i in range(n_points - 1):
            if time.time() > deadline - 3.0:
                break
            fi = grid_f[i]
            fj = grid_f[i + 1]
            
            if not (math.isfinite(fi) and math.isfinite(fj)):
                continue
            
            # Check for sign change
            if fi * fj < 0:
                # Found a bracket [grid_x[i], grid_x[i+1]]
                result = _bisect_cached(cached_f, grid_x[i], grid_x[i+1], fi, fj, 
                                       best_x, best_abs, tol, deadline, cache)
                if result is not None:
                    return result
        
        # If we found a good candidate but no bracket, try Newton
        if best_abs <= 1e-4:
            polished = _newton_polish_cached(cached_f, best_x, best_abs, tol, deadline, cache)
            if polished is not None:
                return polished
    
    # Phase 2: If grid didn't find a bracket, fall back to expanding search
    result = _expanding_search_cached(cached_f, x, best_x, best_abs, tol, deadline, cache)
    
    # Phase 3: High-precision polish with Newton if we have a good candidate
    if best_abs <= 1e-6 and best_abs > tol:
        polished = _newton_polish_cached(cached_f, best_x, best_abs, tol, deadline, cache)
        if polished is not None:
            return polished
    
    return result


def _bisect_cached(cached_f, lo, hi, flo, fhi, best_x, best_abs, tol, deadline, cache):
    """Bisection within [lo, hi] where f(lo)*f(hi) < 0, using cached function."""
    for i in range(80):
        if time.time() > deadline - 2.0:
            break
        mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi:
            break
        fmid = cached_f(mid)
        if not math.isfinite(fmid):
            break
        if abs(fmid) < best_abs:
            best_abs = abs(fmid)
            best_x = mid
        if best_abs <= tol:
            return best_x
        if flo * fmid <= 0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid
    
    return best_x if best_abs <= tol else None


def _newton_polish_cached(cached_f, x, best_abs, tol, deadline, cache):
    """
    High-precision Newton polish from a good starting point, using cached function.
    Returns x if |f(x)| <= tol, else None.
    """
    x = float(x)
    for i in range(30):
        if time.time() > deadline - 1.0:
            return None
        fx = cached_f(x)
        if not math.isfinite(fx):
            return None
        if abs(fx) <= tol:
            return x
        if abs(fx) > best_abs:
            return None
        
        # Use a smaller step for derivative estimation in polish phase
        h = 1e-8 * max(1.0, abs(x))
        fxp = cached_f(x + h)
        if not math.isfinite(fxp):
            return None
        d = (fxp - fx) / h
        if d == 0 or not math.isfinite(d):
            return None
        
        x_new = x - fx / d
        if not math.isfinite(x_new):
            return None
        
        # Conservative step: don't move too far
        max_step = 1e-4 * max(1.0, abs(x))
        if abs(x_new - x) > max_step:
            x_new = x + math.copysign(max_step, x_new - x)
        
        x = x_new
    
    return None


def _expanding_search_cached(cached_f, x0, best_x, best_abs, tol, deadline, cache):
    """
    Expanding search for a sign change, then bisection.
    """
    scale = 1e-3 * max(1.0, abs(x0))
    for attempt in range(20):
        if time.time() > deadline - 3.0:
            break
        lo = x0 - scale
        hi = x0 + scale
        flo = cached_f(lo)
        fhi = cached_f(hi)
        
        if not (math.isfinite(flo) and math.isfinite(fhi)):
            scale *= 2.0
            continue
        
        # Update best
        if abs(flo) < best_abs:
            best_abs = abs(flo)
            best_x = lo
        if abs(fhi) < best_abs:
            best_abs = abs(fhi)
            best_x = hi
        if best_abs <= tol:
            return best_x
        
        if flo * fhi < 0:
            # Found a bracket
            result = _bisect_cached(cached_f, lo, hi, flo, fhi, best_x, best_abs, tol, deadline, cache)
            if result is not None:
                return result
        
        scale *= 2.0
    
    # If no bracket found, try Newton from best_x
    polished = _newton_polish_cached(cached_f, best_x, best_abs, tol, deadline, cache)
    if polished is not None:
        return polished
    
    return best_x
# EVOLVE-BLOCK-END
