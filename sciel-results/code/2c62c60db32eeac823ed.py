# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _union_length_1d(x, w=0.125):
    """
    Compute the total length of the union of intervals [x_i, x_i + w] for i=0..N-1.
    x is a 1D array of base positions.
    """
    n = len(x)
    if n == 0:
        return 0.0
    # Sort the intervals by their left endpoint
    order = np.argsort(x)
    xs = x[order]
    # Merge intervals
    merged = []
    start = xs[0]
    end = xs[0] + w
    for i in range(1, n):
        if xs[i] <= end:
            if xs[i] + w > end:
                end = xs[i] + w
        else:
            merged.append((start, end))
            start = xs[i]
            end = xs[i] + w
    merged.append((start, end))
    total_len = sum(e - s for s, e in merged)
    return total_len

def _cost_1d(x, w=0.125):
    """
    Objective function for 1D optimization: minimize the union length of intervals.
    """
    return _union_length_1d(x, w)

def _gradient_1d(x, w=0.125, eps=1e-6):
    """
    Numerical gradient for the 1D union length objective.
    """
    n = len(x)
    grad = np.zeros(n)
    base = _cost_1d(x, w)
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += eps
        f_plus = _cost_1d(x_plus, w)
        x_minus = x.copy()
        x_minus[i] -= eps
        f_minus = _cost_1d(x_minus, w)
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad

def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    We build N*N = 64 "generalized tubes" indexed by (i, j), i, j in {0, ..., N-1}.
    Tube (i, j) has a base position (x, y) and, at height c in [0, 1], occupies the
    axis-aligned square
        [x + c*i/N, x + 1/N + c*i/N] x [y + c*j/N, y + 1/N + c*j/N].
    
    The union volume is a product of the 1D union lengths in x and y due to the
    product structure. We optimize the 16 base positions (8 in x, 8 in y) to
    minimize the union volume.
    """
    N = 8
    w = 1.0 / N
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 30.0  # seconds
    
    # Initial positions using Keich binary construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    # Try multiple random restarts to find a good local minimum
    best_x = keich.copy()
    best_y = keich.copy()
    best_cost = _cost_1d(best_x, w) * _cost_1d(best_y, w)
    
    rng = np.random.default_rng(42)
    
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit - safety_margin:
            break
            
        # Perturb the initial positions
        if restart == 0:
            x0 = keich.copy()
            y0 = keich.copy()
        else:
            # Random perturbation around the Keich positions
            x0 = keich + rng.normal(0, 0.1, N)
            y0 = keich + rng.normal(0, 0.1, N)
        
        # Optimize x positions
        result_x = minimize(_cost_1d, x0, args=(w,), method='Nelder-Mead',
                           options={'maxiter': 1000, 'fatol': 1e-10, 'xatol': 1e-10})
        if result_x.success or result_x.fun < best_cost:
            x_opt = result_x.x
        
        # Optimize y positions
        result_y = minimize(_cost_1d, y0, args=(w,), method='Nelder-Mead',
                           options={'maxiter': 1000, 'fatol': 1e-10, 'xatol': 1e-10})
        if result_y.success or result_y.fun < best_cost:
            y_opt = result_y.x
        
        # Evaluate the combined cost
        cost = _cost_1d(x_opt, w) * _cost_1d(y_opt, w)
        
        if cost < best_cost:
            best_cost = cost
            best_x = x_opt.copy()
            best_y = y_opt.copy()
    
    # Build the 64x2 position array
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_x[i_flat]
    base_y = best_y[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
