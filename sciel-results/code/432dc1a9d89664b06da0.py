# EVOLVE-BLOCK-START
import numpy as np


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem. For a fixed n, choose
    base positions x = (x_0, ..., x_{n-1}). For position index i (0-based) let
    a_i = (i + 1) / n and define the triangle R_i with vertices
        (x_i, 0), (x_i + 1/n, 0), (x_i + a_i, 1).
    Every triangle has base width 1/n on the line y = 0 and its apex on the line
    y = 1. The objective is to choose x so that the AREA of the union of the n
    triangles R_i is as small as possible. Here n = 64.

    This implementation uses a projected subgradient method with vectorized
    operations to efficiently minimize the union area.

    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    n = 64
    n_m = 20000
    y = (np.arange(n_m) + 0.5) / n_m
    a = (np.arange(1, n + 1)) / n  # a_i = (i+1)/n, i=0..n-1
    w = (1.0 - y) / n  # width at each height, shape (n_m,)

    def compute_area_and_grad(x):
        """Compute the union area and subgradient for given positions x."""
        x = np.asarray(x, dtype=np.float64)
        # L[i, m] = x_i + a_i * y_m, shape (n, n_m)
        L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
        
        # Sort along axis 0 (over i) for each m
        # We need the sorted order and the gaps
        order = np.argsort(L, axis=0)  # shape (n, n_m)
        L_sorted = np.take_along_axis(L, order, axis=0)  # shape (n, n_m)
        gaps = np.diff(L_sorted, axis=0)  # shape (n-1, n_m)
        
        # Union length at each m: w_m + sum_i min(w_m, gaps[i, m])
        min_gaps = np.minimum(w[np.newaxis, :], gaps)  # shape (n-1, n_m)
        union_lengths = w + np.sum(min_gaps, axis=0)  # shape (n_m,)
        area = np.mean(union_lengths)
        
        # Subgradient: for each m, for each i where gaps[i, m] < w_m,
        # the subgradient contribution is -1 for the lower index and +1 for the higher index
        # in the sorted order.
        mask = gaps < w[np.newaxis, :]  # shape (n-1, n_m)
        
        # Build gradient: grad[i] = sum over m of (contribution from i)
        # For each m and each consecutive pair (i, i+1) in sorted order, if mask[i, m],
        # then grad[order[i, m]] -= 1 and grad[order[i+1, m]] += 1
        grad = np.zeros(n)
        
        # Vectorized approach:
        # For each m, the indices in sorted order are order[:, m]
        # For each i in 0..n-2, if mask[i, m], add -1 to grad[order[i, m]] and +1 to grad[order[i+1, m]]
        
        # This is hard to vectorize directly. Let's do it per m but with vectorized inner ops.
        # Actually, let's use a different approach: accumulate using np.add.at
        lower_indices = order[:-1, :]  # shape (n-1, n_m)
        upper_indices = order[1:, :]   # shape (n-1, n_m)
        
        # Flatten for np.add.at
        lower_flat = lower_indices[mask]  # indices where mask is True
        upper_flat = upper_indices[mask]
        
        np.add.at(grad, lower_flat, -1.0)
        np.add.at(grad, upper_flat, 1.0)
        
        grad /= n_m
        return area, grad

    # Initialize with Keich construction
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    x = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    
    best_x = x.copy()
    best_area, _ = compute_area_and_grad(x)
    
    # Subgradient method with adaptive step size
    step_size = 0.01
    for it in range(2000):
        area, grad = compute_area_and_grad(x)
        x_new = x - step_size * grad
        area_new, _ = compute_area_and_grad(x_new)
        if area_new < best_area:
            best_area = area_new
            best_x = x_new.copy()
            x = x_new
            step_size *= 0.99
        else:
            step_size *= 0.95
        if step_size < 1e-8:
            break
    
    return best_x.astype(np.float64)


# EVOLVE-BLOCK-END
