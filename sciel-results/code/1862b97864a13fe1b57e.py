# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_area(x):
    """
    Compute the area of the union of the 64 triangles.
    x: array of shape (64,)
    """
    n = 64
    # y_m = (m + 0.5) / 20000 for m = 0..19999
    m = np.arange(20000)
    y = (m + 0.5) / 20000.0
    w = (1.0 - y) / n  # shape (20000,)
    
    # a_i = (i + 1) / 64
    a = (np.arange(1, n + 1)) / n  # shape (64,)
    
    # L_i = x_i + a_i * y
    # x is (64,), a is (64,), y is (20000,)
    # We want L to be shape (64, 20000)
    # L[i, m] = x[i] + a[i] * y[m]
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    
    # Sort L along axis 0 (for each y, sort the L_i values)
    L_sorted = np.sort(L, axis=0)
    
    # Union length calculation:
    # Union length = w + sum_{j=0}^{n-2} min(L_sorted[j+1] - L_sorted[j], w)
    # Note: The intervals are [L_i, L_i + w].
    # If we sort the starting points L_i, the union length is the length of the 
    # connected components.
    # The formula given in the prompt: w + sum over consecutive pairs of min(L_{j+1} - L_j, w).
    # This assumes L_0 is the smallest. The first interval covers [L_0, L_0+w].
    # The next interval starts at L_1. The overlap is max(0, L_0+w - L_1).
    # The new length added is (L_1+w) - max(L_0, L_1) ... wait.
    # Let's check the formula: w + sum min(diff, w).
    # If diff < w, they overlap. The union length increases by diff.
    # If diff >= w, they are disjoint. The union length increases by w.
    # So the total length is w (for the first interval) + sum_{j=1}^{n-1} min(L_j - L_{j-1}, w).
    # This matches the prompt.
    
    diffs = np.diff(L_sorted, axis=0)  # shape (63, 20000)
    # min(diffs, w)
    # w is shape (20000,), needs broadcasting
    added_lengths = np.minimum(diffs, w[np.newaxis, :])
    
    # Union length for each y is w[m] + sum(added_lengths[:, m])
    union_lengths = w + np.sum(added_lengths, axis=0)
    
    # Area is mean of union lengths
    area = np.mean(union_lengths)
    return area

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    # Initial positions: Keich construction
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x = np.array(positions, dtype=np.float64)
    
    current_area = _compute_area(x)
    
    # Coordinate Descent with Ternary Search
    start_time = time.time()
    time_limit = 500.0  # Leave some margin for overhead
    
    # Ternary search parameters
    # The optimal x_i is likely within a small range of the current value.
    # The scale of x is roughly 0 to 1.
    # Let's search in [x_i - 0.5, x_i + 0.5] initially, then narrow down.
    # Or just use a fixed small range if we know it's local.
    # Given the convexity, we can start with a wide range and shrink.
    
    # To be safe and fast, we'll do a few passes of coordinate descent.
    # In each pass, for each i, we perform ternary search on [x_i - 1.0, x_i + 1.0].
    # This is safe because the triangles are within [0,1] roughly.
    
    num_iterations = 0
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            # Ternary search for x[i]
            lo = x[i] - 1.0
            hi = x[i] + 1.0
            tol = 1e-6
            while hi - lo > tol:
                m1 = lo + (hi - lo) / 3.0
                m2 = hi - (hi - lo) / 3.0
                x_test1 = x.copy()
                x_test1[i] = m1
                f1 = _compute_area(x_test1)
                
                x_test2 = x.copy()
                x_test2[i] = m2
                f2 = _compute_area(x_test2)
                
                if f1 < f2:
                    hi = m2
                else:
                    lo = m1
            new_xi = (lo + hi) / 2.0
            x_test = x.copy()
            x_test[i] = new_xi
            f_new = _compute_area(x_test)
            
            if f_new < current_area:
                x[i] = new_xi
                current_area = f_new
                improved = True
        
        # If no improvement in a full pass, we might be in a local minimum or converged.
        # We can try to widen the search or add noise.
        if not improved:
            # Add small random perturbation to escape local minima
            # Since it's convex per coordinate, local minima are global for that coordinate,
            # but the joint function might have issues? No, coordinate descent on 
            # convex-per-coordinate functions converges to a stationary point.
            # If no improvement, it's likely converged.
            break
        num_iterations += 1
        if num_iterations > 20:
            break

    return x

# EVOLVE-BLOCK-END
