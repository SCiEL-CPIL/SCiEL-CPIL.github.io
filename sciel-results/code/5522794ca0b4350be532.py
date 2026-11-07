import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 118.0  # Safety margin for 120s limit
    
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

    # Initial evaluation
    x0 = float(x0)
    fx0 = _f(x0)
    
    if abs(fx0) <= 1e-8:
        return x0

    # Strategy: Global Minimization of |f(x)|
    # 1. Multi-scale sampling to find the global minimum of |f(x)|.
    #    This avoids the "missing the root" problem of bracketing when
    #    the grid step is larger than the oscillation period.
    # 2. Once the global minimum is located, perform a local Golden Section
    #    Search around that point to refine |f(x)| to <= 1e-8.
    
    # We will generate a set of candidate points to evaluate.
    # The points are chosen to cover a wide range with varying densities.
    # Density is highest near x0.
    
    candidates = []
    
    # Helper to add a range of points
    def add_range(start, end, num_points):
        if num_points <= 0:
            return
        if num_points == 1:
            candidates.append(start)
            return
        step = (end - start) / (num_points - 1)
        for i in range(num_points):
            candidates.append(start + i * step)
            
    # Define grids around x0
    # We want to balance coverage and density.
    # Total budget for sampling: ~200 calls.
    # Refinement budget: ~250 calls.
    
    # Grid 1: Very local, high density. Range [-1, 1], 40 points. Step ~0.05
    add_range(x0 - 1.0, x0 + 1.0, 40)
    
    # Grid 2: Local, medium density. Range [-5, 5], 40 points. Step ~0.25
    add_range(x0 - 5.0, x0 + 5.0, 40)
    
    # Grid 3: Medium, lower density. Range [-20, 20], 40 points. Step ~1
    add_range(x0 - 20.0, x0 + 20.0, 40)
    
    # Grid 4: Wide, low density. Range [-100, 100], 40 points. Step ~5
    add_range(x0 - 100.0, x0 + 100.0, 40)
    
    # Grid 5: Very wide, very low density. Range [-500, 500], 20 points. Step ~50
    add_range(x0 - 500.0, x0 + 500.0, 20)
    
    # Grid 6: Huge, sparse. Range [-2000, 2000], 10 points. Step ~400
    add_range(x0 - 2000.0, x0 + 2000.0, 10)
    
    # Grid 7: Extreme, sparse. Range [-10000, 10000], 5 points.
    add_range(x0 - 10000.0, x0 + 10000.0, 5)
    
    # Remove duplicates and sort by distance from x0 to prioritize local search
    # Sorting by distance ensures we find the local min first, which is often the root
    # if the function is well-behaved locally. If the global min is far, we'll find it
    # as we expand.
    candidates = list(set(candidates))
    candidates.sort(key=lambda x: abs(x - x0))
    
    # Evaluate candidates
    # We stop if we find a root or if we've used our sampling budget.
    # Sampling budget: 250 calls.
    
    for x in candidates:
        if calls >= 250:
            break
        if time.time() - start_time > TIME_LIMIT:
            break
            
        fx = _f(x)
        
        # If we hit a root directly, return
        if abs(fx) <= 1e-8:
            return x
            
    # Now, best_x holds the location of the global minimum of |f(x)| found so far.
    # best_f_abs holds the value.
    
    # If best_f_abs is already small enough, return.
    if best_f_abs <= 1e-8:
        return best_x
        
    # Refinement Phase: Golden Section Search
    # We define a local interval around best_x.
    # The width of the interval should be comparable to the grid spacing
    # that produced best_x, or slightly larger to ensure we bracket the local min.
    
    # Estimate the local scale.
    # If best_x is from Grid 1 (range 2, 40 pts), step ~ 0.05.
    # If from Grid 2 (range 10, 40 pts), step ~ 0.25.
    # We can just use a fixed reasonable width, e.g., 0.1, and expand if needed.
    # Or better, determine the width based on which grid point it came from?
    # Simpler: Start with a width of 0.5. If the min is at the boundary, expand.
    
    # Let's use a width of 1.0 initially. This is safe for most local minima.
    # If the function is extremely oscillatory (period < 0.1), our grid might have
    # missed the true min, but the "walking" logic in the previous inspiration suggested
    # that checking intermediate points is good. However, for minimization, if we miss
    # the root, we find a local min that is not a root.
    
    # To mitigate missing fast oscillations, we can do a "dense check" around best_x
    # before golden section.
    
    # Let's refine using Golden Section Search on [best_x - width, best_x + width]
    # We want to minimize g(x) = |f(x)|.
    
    width = 0.5
    # It's possible the global min is at the edge of our search range or the local min
    # is narrower than width.
    
    # Let's do a few iterations of "expand if min is at boundary".
    # But first, let's just run Golden Section on [best_x - width, best_x + width].
    
    a = best_x - width
    b = best_x + width
    
    # Ensure we evaluate endpoints if not already done?
    # _f handles caching.
    fa = _f(a)
    fb = _f(b)
    
    # Golden Section Search for minimum of |f(x)|
    # We need to be careful: g(x) = |f(x)| is not differentiable at roots,
    # but Golden Section works for unimodal functions. |f(x)| is unimodal
    # in a small neighborhood of a simple root.
    
    # Constants for Golden Section
    gr = (math.sqrt(5) - 1) / 2  # ~0.618
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    
    fc = _f(c)
    fd = _f(d)
    
    # We minimize abs(f(x))
    # Note: _f returns f(x), we need abs(f(x)) for comparison
    # But _f updates best_x/best_f_abs based on abs.
    # We should compare abs(fc) and abs(fd).
    
    for _ in range(100):
        if calls >= 480:
            break
        if time.time() - start_time > TIME_LIMIT:
            break
            
        if abs(fc) <= 1e-8:
            return c
        if abs(fd) <= 1e-8:
            return d
            
        if abs(fc) < abs(fd):
            # Min is in [a, d]
            b = d
            fb = fd
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = _f(c)
        else:
            # Min is in [c, b]
            a = c
            fa = fc
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = _f(d)
            
        # Check for convergence or bracket loss?
        # If the interval becomes too small, stop.
        if abs(b - a) < 1e-12:
            break
            
    # Return the best point found
    # Check c and d and best_x
    candidates_refine = [best_x, c, d, a, b]
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
