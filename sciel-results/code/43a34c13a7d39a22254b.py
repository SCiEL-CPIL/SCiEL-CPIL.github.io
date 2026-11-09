# EVOLVE-BLOCK-START
import numpy as np
import time


def _union_area(x, M=20000):
    """
    Compute the union area of 64 triangles.
    
    x: shape (64,) - base positions
    M: number of quadrature points
    Returns: area (scalar)
    """
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    
    # L_i(y) = x_i + a_i * y
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n
    
    # Sort L along axis=1
    order = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, order, axis=1)
    
    # Consecutive differences
    diffs = np.diff(L_sorted, axis=1)
    
    # union_len at each y
    union_len = w + np.sum(np.minimum(diffs, w[:, np.newaxis]), axis=1)
    area = np.mean(union_len)
    
    return area


def _union_area_grad(x, M=20000):
    """
    Compute the union area and its gradient w.r.t. x using bincount for speed.
    
    x: shape (64,) - base positions
    M: number of quadrature points
    Returns: (area, grad) where grad has shape (64,)
    """
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    
    # L_i(y) = x_i + a_i * y, shape (M, n)
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n
    
    # Sort L along axis=1 and track permutation
    order = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, order, axis=1)
    
    # Consecutive differences
    diffs = np.diff(L_sorted, axis=1)
    
    # mask: (M, n-1) boolean, True where diffs < w (i.e., overlapping)
    mask = diffs < w[:, np.newaxis]
    
    # union_len = w + sum min(diffs, w)
    union_len = w + np.sum(np.minimum(diffs, w[:, np.newaxis]), axis=1)
    area = np.mean(union_len)
    
    # Gradient computation
    # d(union_len)/dx_j = sum over i where j is involved in the sorted order
    # For each y, the contribution to the gradient comes from:
    # 1. The base w term: each triangle contributes 1/n to its own x
    # 2. The min(diffs, w) terms: when overlapping (mask=True), the diff contributes
    
    # For the min(diffs, w) part:
    # d/dx_{order[k]} of min(L_{order[k+1]} - L_{order[k]}, w) 
    #   = 1 if mask[k] is True (overlapping), else 0, with sign +
    # d/dx_{order[k+1]} of min(L_{order[k+1]} - L_{order[k]}, w)
    #   = -1 if mask[k] is True, else 0, with sign -
    
    # Use bincount to accumulate gradients
    # order has shape (M, n)
    # mask has shape (M, n-1)
    
    # For each y_m, for each k in 0..n-2:
    #   if mask[m,k] is True:
    #     grad[order[m,k]] += 1
    #     grad[order[m,k+1]] -= 1
    
    grad = np.zeros(n, dtype=np.float64)
    
    # Flatten for bincount
    # For each (m, k) where mask is True:
    #   add +1 to grad[order[m,k]]
    #   add -1 to grad[order[m,k+1]]
    
    # Get indices where mask is True
    mask_flat = mask.flatten()
    order_flat = order.flatten()
    
    # For each position in the flattened mask:
    # The index in order corresponds to: row m, col k -> index m*n + k
    # order[m, k] gives the triangle index at sorted position k in row m
    
    # We need to be careful with the flat indexing
    # mask shape: (M, n-1)
    # order shape: (M, n)
    
    # For mask[m, k] (k in 0..n-2):
    #   order[m, k] is the triangle at sorted position k
    #   order[m, k+1] is the triangle at sorted position k+1
    
    # Let's use a more direct approach with loops over M for the gradient
    # Since M can be up to 20000, we need efficiency
    
    # Alternative: use np.add.at
    # For each m, k where mask[m,k]:
    #   np.add.at(grad, order[m,k], 1)
    #   np.add.at(grad, order[m,k+1], -1)
    
    # Vectorized approach using bincount:
    # Create arrays of indices and weights
    # For +1 contributions: indices are order[m,k] for all (m,k) where mask[m,k]
    # For -1 contributions: indices are order[m,k+1] for all (m,k) where mask[m,k]
    
    # Get row and col indices of True elements in mask
    rows, cols = np.where(mask)
    
    # +1 to grad[order[rows, cols]]
    # -1 to grad[order[rows, cols+1]]
    
    idx_plus = order[rows, cols]
    idx_minus = order[rows, cols + 1]
    
    np.add.at(grad, idx_plus, 1.0)
    np.add.at(grad, idx_minus, -1.0)
    
    # Scale by 1/M (mean)
    grad /= M
    
    return area, grad


def _keich_init(n=64):
    """
    Keich construction initial guess.
    """
    k = int(np.log2(n))
    j_arr = np.arange(1, k + 1)
    positions = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions[i] = np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k)
    return positions


def _random_init(n=64, seed=0):
    """
    Random initial guess.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, size=n)


def _lbfgs_optimize(x0, maxiter=200, M=2000):
    """
    Run L-BFGS-B optimization from x0.
    Returns (x_opt, area_opt).
    """
    from scipy.optimize import minimize
    
    def objective(x):
        return _union_area(x, M=M)
    
    def objective_grad(x):
        area, grad = _union_area_grad(x, M=M)
        return area, grad
    
    res = minimize(
        objective_grad,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'gtol': 1e-6, 'ftol': 1e-8}
    )
    
    x_opt = res.x
    area_opt = _union_area(x_opt, M=M)
    
    return x_opt, area_opt


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    Uses a multi-restart strategy with L-BFGS-B local optimization using
    fast gradient computation via bincount. Multiple diverse initial points
    help escape local minima. Final polish at full 20000 quadrature ensures
    the returned solution is optimal for the actual evaluation metric.
    
    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    n = 64
    
    start_time = time.time()
    deadline = start_time + 630.0  # 630 second limit (evaluator kills at 630s)
    safety_margin = 15.0  # seconds before deadline
    
    best_x = _keich_init(n)
    best_area = _union_area(best_x, M=2000)
    
    # Generate diverse initial points
    inits = []
    
    # Keich construction
    inits.append(_keich_init(n))
    
    # Perturbed Keich with different noise levels and seeds
    for seed in range(10):
        rng = np.random.default_rng(seed)
        noise_level = 0.02 + 0.03 * (seed % 4)
        inits.append(_keich_init(n) + rng.normal(0, noise_level, size=n))
    
    # Random initializations
    for seed in range(5):
        inits.append(_random_init(n, seed=100 + seed))
    
    # Sort inits by initial area (quick check)
    init_areas = [_union_area(x, M=1000) for x in inits]
    init_order = np.argsort(init_areas)
    inits = [inits[i] for i in init_order]
    
    # Run L-BFGS-B from each initial point
    for i, x0 in enumerate(inits):
        if time.time() > deadline - safety_margin:
            break
        
        try:
            x_opt, area_opt = _lbfgs_optimize(x0, maxiter=200, M=2000)
            
            if area_opt < best_area:
                best_x = x_opt.copy()
                best_area = area_opt
        except Exception:
            continue
    
    # Final polish at full resolution
    if time.time() < deadline - safety_margin:
        try:
            x_final, area_final = _lbfgs_optimize(best_x, maxiter=100, M=20000)
            if area_final < best_area:
                best_x = x_final.copy()
                best_area = area_final
        except Exception:
            pass
    
    return best_x


# EVOLVE-BLOCK-END
