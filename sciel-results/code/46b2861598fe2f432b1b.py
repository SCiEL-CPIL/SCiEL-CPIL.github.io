# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_area_and_grad(x, n=64, M=20000):
    """Compute area and gradient of union area using vectorized operations."""
    a = (np.arange(n) + 1) / n  # a_i = (i+1)/n
    w0 = 1.0 / n
    y = (np.arange(M) + 0.5) / M  # M quadrature points
    
    # L_i(y) = x_i + a_i * y, shape (n, M)
    L = x[:, None] + a[:, None] * y[None, :]
    w = (1.0 - y[None, :]) * w0  # width at each y, shape (M,)
    
    # Sort L at each y
    idx = np.argsort(L, axis=0)  # (n, M)
    L_sorted = np.take_along_axis(L, idx, axis=0)  # (n, M)
    
    # Gaps between consecutive sorted L values
    gaps = np.diff(L_sorted, axis=0)  # (n-1, M)
    
    # Union length = w + sum(min(gap, w))
    union_lengths = w + np.minimum(gaps, w[None, :]).sum(axis=0)  # (M,)
    area = np.mean(union_lengths)
    
    # Gradient: d(area)/d(x_i)
    # For each y, the contribution to union length from gap k (between sorted positions k and k+1)
    # is min(gap_k, w). The derivative w.r.t. x_i is:
    #   +1 if x_i is the left endpoint of a gap where gap < w (i.e., the gap is "active")
    #   -1 if x_i is the right endpoint of such a gap
    #   0 otherwise (gap >= w means the triangles are separated, no overlap contribution)
    
    # For each y, find which gaps are "active" (gap < w)
    active = gaps < w[None, :]  # (n-1, M), True where gap < w
    
    # For each active gap at position k (between sorted[k] and sorted[k+1]),
    # the left element contributes +1 and right contributes -1 to the gradient
    # We need to map back to original indices
    
    # grad[i, m] = sum over gaps k where active[k,m]:
    #   +1 if idx[k, m] == i (i is left endpoint)
    #   -1 if idx[k+1, m] == i (i is right endpoint)
    
    # Vectorized: for each gap k, add +1 to row idx[k,m] and -1 to row idx[k+1,m]
    # Use bincount for efficiency
    
    grad = np.zeros((n, M), dtype=np.float64)
    
    # For each gap position k (0 to n-2), and each m:
    # If active[k, m], then grad[idx[k,m], m] += 1 and grad[idx[k+1,m], m] -= 1
    
    # Flatten for bincount: create arrays of indices and weights
    # For left endpoints:
    left_idx = idx[:-1, :]  # (n-1, M)
    right_idx = idx[1:, :]  # (n-1, M)
    
    # Weight: 1 where active, 0 otherwise
    weight = active.astype(np.float64)  # (n-1, M)
    
    # For each column m, sum contributions
    # left contributions: for each k, if active[k,m], add 1 to grad[left_idx[k,m], m]
    # right contributions: for each k, if active[k,m], subtract 1 from grad[right_idx[k,m], m]
    
    # Use np.add.at or bincount per column - but that's slow for M=20000
    # Alternative: use scatter with repeated indices
    
    # Flatten: for left, we have (n-1)*M entries
    left_flat = left_idx.ravel()  # (n-1)*M
    right_flat = right_idx.ravel()  # (n-1)*M
    weight_flat = weight.ravel()  # (n-1)*M
    
    # Column index for each entry: entry j corresponds to m = j % M
    col_idx = np.arange((n-1) * M) % M  # (n-1)*M
    
    # This is getting complex. Let's use a different approach:
    # For each m, compute the gradient by processing the sorted array
    
    # Actually, let's use a simpler but still vectorized approach:
    # grad[i, m] = (number of active gaps where i is the left endpoint) - (number where i is right endpoint)
    
    # For left endpoint: count active gaps k where idx[k, m] == i
    # For right endpoint: count active gaps k where idx[k+1, m] == i
    
    # Create a 2D array: for each i and m, count
    # This is O(n*M) with sorting already done
    
    # Let's just do it with a loop over n (64 iterations, each O(M))
    grad = np.zeros((n, M), dtype=np.float64)
    for i in range(n):
        # Count active gaps where i is left endpoint
        left_count = np.sum((idx[:-1, :] == i) & active, axis=0)  # (M,)
        # Count active gaps where i is right endpoint
        right_count = np.sum((idx[1:, :] == i) & active, axis=0)  # (M,)
        grad[i, :] = left_count - right_count
    
    # The gradient of area w.r.t. x_i is the mean of grad[i, :] over m
    # because area = (1/M) * sum_m union_length(m)
    # and d(union_length(m))/d(x_i) = grad[i, m]
    
    grad_x = grad.mean(axis=1)  # (n,)
    
    return area, grad_x


def _compute_area_fast(x, n=64, M=20000):
    """Compute only the area (no gradient) for evaluation."""
    a = (np.arange(n) + 1) / n
    w0 = 1.0 / n
    y = (np.arange(M) + 0.5) / M
    
    L = x[:, None] + a[:, None] * y[None, :]
    w = (1.0 - y[None, :]) * w0
    
    idx = np.argsort(L, axis=0)
    L_sorted = np.take_along_axis(L, idx, axis=0)
    gaps = np.diff(L_sorted, axis=0)
    
    union_lengths = w + np.minimum(gaps, w[None, :]).sum(axis=0)
    area = np.mean(union_lengths)
    return area


def _keich_initial(n=64):
    """Generate the Keich construction initial point."""
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    Uses multi-restart L-BFGS-B optimization with fast gradient computation
    to find base positions that minimize the union area of 64 triangles.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit
    
    n = 64
    M = 20000  # Full quadrature resolution
    
    # Generate diverse initial points
    x_keich = _keich_initial(n)
    
    # Create multiple initializations
    inits = [x_keich.copy()]
    
    # Add perturbed versions of Keich
    rng = np.random.default_rng(42)
    for trial in range(10):
        scale = 0.01 * (trial + 1)
        x_pert = x_keich + rng.normal(0, scale, size=n)
        inits.append(x_pert)
    
    # Add a few random initializations
    for trial in range(5):
        x_rand = rng.uniform(-0.5, 0.5, size=n)
        inits.append(x_rand)
    
    # Evaluate each initialization and run optimization
    best_x = x_keich.copy()
    best_area = _compute_area_fast(x_keich, n, M)
    
    for init_idx, x0 in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > deadline - 60:  # Leave 60s margin
            break
        
        # Run L-BFGS-B
        try:
            result = minimize(
                fun=lambda x: _compute_area_and_grad(x, n, M)[0],
                x0=x0,
                jac=lambda x: _compute_area_and_grad(x, n, M)[1],
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_area:
                # Evaluate at full resolution to confirm
                area_full = _compute_area_fast(result.x, n, M)
                if area_full < best_area:
                    best_area = area_full
                    best_x = result.x.copy()
        except Exception:
            continue
    
    # Final polish with more iterations if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        try:
            result = minimize(
                fun=lambda x: _compute_area_and_grad(x, n, M)[0],
                x0=best_x,
                jac=lambda x: _compute_area_and_grad(x, n, M)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result.fun < best_area:
                area_full = _compute_area_fast(result.x, n, M)
                if area_full < best_area:
                    best_area = area_full
                    best_x = result.x.copy()
        except Exception:
            pass
    
    # Ensure valid output
    if not np.all(np.isfinite(best_x)):
        best_x = x_keich
    
    return best_x.astype(np.float64)


# EVOLVE-BLOCK-END
