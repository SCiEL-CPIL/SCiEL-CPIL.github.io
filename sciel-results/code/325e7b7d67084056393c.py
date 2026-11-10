# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_area(x, n=64, n_heights=20000):
    """Compute the union area of n triangles with base positions x."""
    a = (np.arange(n) + 1) / n
    y = (np.arange(n_heights) + 0.5) / n_heights
    w = (1.0 - y) / n
    # L[i, m] = x_i + a_i * y_m
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    # Sort L along axis 0 at each height
    L_sorted = np.sort(L, axis=0)
    # Gaps between consecutive L values
    gaps = L_sorted[1:, :] - L_sorted[:-1, :]
    # Union length = w + sum(min(gap, w))
    union_len = w[np.newaxis, :] + np.sum(np.minimum(gaps, w[np.newaxis, :]), axis=0)
    return np.mean(union_len)


def _local_search(x0, n=64, max_iter=2000, deadline=None):
    """Local search to minimize area. Uses coordinate descent with adaptive step size."""
    x = x0.copy()
    best_area = _compute_area(x, n)
    step_size = 0.01
    n = 64
    a = (np.arange(n) + 1) / n
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            # Try perturbations
            for sign in [1, -1]:
                x_new = x.copy()
                x_new[i] += sign * step_size
                area = _compute_area(x_new, n)
                if area < best_area - 1e-12:
                    x = x_new
                    best_area = area
                    improved = True
                    break
            if not improved:
                # Try smaller step for this coordinate
                for sign in [1, -1]:
                    x_new = x.copy()
                    x_new[i] += sign * step_size * 0.5
                    area = _compute_area(x_new, n)
                    if area < best_area - 1e-12:
                        x = x_new
                        best_area = area
                        improved = True
                        break
        if not improved:
            step_size *= 0.7
            if step_size < 1e-10:
                break
    return x, best_area


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    Uses local search from multiple initializations (Keich, aligned) to find
    a low-area configuration.
    """
    n = 64
    deadline = time.time() + 580  # Safety margin before 630s limit
    
    # Keich construction
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    keich = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        keich[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    
    # Aligned initialization: x_i = -a_i + 0.5
    a = (np.arange(n) + 1) / n
    aligned = -a + 0.5
    
    # Try multiple initializations
    best_x = keich
    best_area = _compute_area(keich, n)
    
    for init in [aligned, keich]:
        if time.time() > deadline:
            break
        x_opt, area_opt = _local_search(init, n, max_iter=2000, deadline=deadline)
        if area_opt < best_area:
            best_x = x_opt
            best_area = area_opt
    
    # Final polish with very small steps
    if time.time() < deadline:
        x_polish, area_polish = _local_search(best_x, n, max_iter=500, deadline=deadline)
        if area_polish < best_area:
            best_x = x_polish
            best_area = area_polish
    
    return best_x.astype(np.float64)


# EVOLVE-BLOCK-END
