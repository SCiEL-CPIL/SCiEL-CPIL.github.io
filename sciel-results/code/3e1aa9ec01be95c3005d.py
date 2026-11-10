# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _calc_area(x):
    """
    Compute the area of the union of the 64 triangles for given positions x.
    n = 64
    """
    n = 64
    w = 1.0 / n
    
    # a_i = (i + 1) / 64
    a = (np.arange(n) + 1) / n
    
    # y_m = (m + 0.5) / 20000, m = 0..19999
    m = np.arange(20000)
    y = (m + 0.5) / 20000.0
    
    # L_i = x_i + a_i * y
    # Shape: (n, 20000)
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    
    # Sort L_i for each y_m
    L_sorted = np.sort(L, axis=0)
    
    # Calculate union length at each height
    # Union length = w + sum over consecutive pairs of min(L_{j+1} - L_j, w)
    # Note: The formula in the prompt is: w + sum min(L_{j+1} - L_j, w)
    # This works because the first triangle contributes w, and each subsequent
    # overlapping part adds min(gap, w). Wait, let's re-read carefully.
    # "union length is w + sum over consecutive pairs of min(L_{j+1} - L_j, w)"
    # This implies the first interval is [L_0, L_0+w].
    # The next interval starts at L_1.
    # The length added is min(L_1 + w - (L_0 + w), w) = min(L_1 - L_0, w) if L_1 > L_0.
    # Since L is sorted, L_{j+1} >= L_j.
    # If L_{j+1} - L_j <= w, the new interval overlaps or touches, adding (L_{j+1} - L_j) to the span?
    # No, if they overlap, the union length increases by the non-overlapping part.
    # Let's trace:
    # Interval 1: [L_0, L_0+w]. Length w.
    # Interval 2: [L_1, L_1+w].
    # If L_1 >= L_0 + w, no overlap. Union length = w + w = 2w.
    #   Formula: w + min(L_1 - L_0, w). If L_1 - L_0 >= w, min is w. Total 2w. Correct.
    # If L_1 < L_0 + w, overlap. Union length = w + (L_1 + w) - (L_0 + w) = w + L_1 - L_0.
    #   Formula: w + min(L_1 - L_0, w). If L_1 - L_0 < w, min is L_1 - L_0. Total w + L_1 - L_0. Correct.
    
    diffs = np.diff(L_sorted, axis=0)
    # diffs shape: (n-1, 20000)
    added_lengths = np.minimum(diffs, w)
    
    # Sum over triangles (axis 0) and add the base width w
    union_lengths = w + np.sum(added_lengths, axis=0)
    
    # Area is the mean of union lengths
    area = np.mean(union_lengths)
    return area

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Optimize the base positions x for the Kakeya needle problem (n=64)
    using Nelder-Mead optimization starting from the Keich construction.
    """
    n = 64
    k = int(np.log2(n))
    
    # 1. Generate Keich initial guess
    j = np.arange(1, k + 1)
    keich_positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        keich_positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x0 = np.array(keich_positions, dtype=np.float64)
    
    # Evaluate initial area
    initial_area = _calc_area(x0)
    
    # Deadline guard
    start_time = time.time()
    timeout = 500.0  # Leave margin for 600s limit
    
    best_x = x0.copy()
    best_area = initial_area
    
    # 2. Optimization
    # Use Nelder-Mead as it doesn't require gradients and is robust for small n
    # maxiter and maxfev are set high, but we will check time manually
    # However, scipy's minimize doesn't natively support time-based stopping easily
    # without wrapping. We can wrap the objective function to raise an exception
    # or return a large value if time is exceeded, but checking inside the loop
    # is safer if we use a callback or just monitor.
    # Since n=64 is small, Nelder-Mead is fast. Let's try a reasonable number of iterations.
    
    options = {
        'maxiter': 2000,
        'maxfev': 2000,
        'xatol': 1e-10,
        'fatol': 1e-10,
        'disp': False
    }
    
    try:
        res = minimize(_calc_area, x0, method='Nelder-Mead', options=options)
        
        if res.success or res.fun < best_area:
            # Check if result is valid (finite)
            if np.all(np.isfinite(res.x)):
                best_x = res.x
                best_area = res.fun
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    
    # If we have spare time, we could do more, but Nelder-Mead on 64 dims
    # with 2000 function evaluations (each involving 20000x64 sort) might be slow.
    # Let's estimate:
    # _calc_area cost:
    # L creation: 64*20000 ops
    # Sort: 20000 * 64 log 64 ~ 20000 * 64 * 6 ~ 7.6M ops
    # Sum/Min: ~1.2M ops
    # Total ~ 10M flops per eval.
    # 2000 evals = 20G flops.
    # NumPy is fast, but Python overhead in minimize loop adds up.
    # 2000 evals might take 10-30 seconds.
    # We have 500s budget. We can afford many more iterations or restarts.
    
    # Let's do multiple restarts with small perturbations if time permits
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < timeout - 10:
        # Perturb the best solution found so far
        # Small Gaussian noise
        x_perturbed = best_x + rng.normal(0, 0.01, n)
        
        # Local optimization from perturbed start
        try:
            res_pert = minimize(_calc_area, x_perturbed, method='Nelder-Mead', options=options)
            if res_pert.success or res_pert.fun < best_area:
                if np.all(np.isfinite(res_pert.x)):
                    best_x = res_pert.x
                    best_area = res_pert.fun
        except Exception:
            pass
        
        # If no improvement in last few iterations, break?
        # For simplicity, just run until time limit.
        
    return best_x

# EVOLVE-BLOCK-END
