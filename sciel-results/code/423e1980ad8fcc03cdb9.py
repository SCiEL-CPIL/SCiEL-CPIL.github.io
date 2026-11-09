# EVOLVE-BLOCK-START
import numpy as np
import time


def _union_area_and_grad(x):
    """Compute union area and gradient using vectorized bincount."""
    n = 64
    a = (np.arange(n) + 1) / n  # a_i = (i+1)/64
    M = 20000
    y = (np.arange(M) + 0.5) / M

    # L_i(y) = x_i + a_i * y, shape (M, n)
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n  # shape (M,)

    # Sort L along axis=1
    order = np.argsort(L, axis=1)  # (M, n)
    L_sorted = np.take_along_axis(L, order, axis=1)
    diffs = np.diff(L_sorted, axis=1)  # (M, n-1)

    # Active pairs: diff < w
    active = (diffs < w[:, np.newaxis]).astype(np.float64)  # (M, n-1)

    # Union length
    union_len = w + np.sum(active * diffs, axis=1)
    area = np.mean(union_len)

    # Gradient via bincount
    # For each active pair (m, j):
    #   grad[right_idx[m,j]] += active[m,j] * a[right_idx[m,j]] / M
    #   grad[left_idx[m,j]]  -= active[m,j] * a[left_idx[m,j]] / M

    left_idx = order[:, :-1]   # (M, n-1)
    right_idx = order[:, 1:]   # (M, n-1)

    # Flatten
    l_flat = left_idx.ravel()
    r_flat = right_idx.ravel()
    w_flat = (active / M).ravel()

    # Contributions
    a_r = a[r_flat]
    a_l = a[l_flat]

    # Use bincount for efficient accumulation
    # grad[r_flat] += w_flat * a_r
    # grad[l_flat] -= w_flat * a_l
    pos_contrib = w_flat * a_r
    neg_contrib = -w_flat * a_l

    grad = np.bincount(r_flat, weights=pos_contrib, minlength=n) + \
           np.bincount(l_flat, weights=neg_contrib, minlength=n)

    return area, grad


def _lbfgs_step(x, maxiter=200, gtol=1e-10):
    """Run L-BFGS-B style optimization manually using scipy."""
    from scipy.optimize import minimize
    result = minimize(
        lambda v: _union_area_and_grad(v)[0],
        x,
        jac=lambda v: _union_area_and_grad(v)[1],
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'gtol': gtol, 'ftol': 1e-14}
    )
    if result.x is not None and np.all(np.isfinite(result.x)):
        return result.x
    return x


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Minimize the union area of 64 Kakeya needle triangles.
    Uses multiple random restarts with L-BFGS-B optimization.
    """
    n = 64
    k = int(np.log2(n))

    # Keich construction as baseline
    j = np.arange(1, k + 1)
    keich = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        keich.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x0_keich = np.array(keich, dtype=np.float64)

    t_start = time.time()
    deadline = t_start + 580.0  # safety margin

    best_x = x0_keich.copy()
    best_area = _union_area_and_grad(best_x)[0]

    # Generate diverse initial points
    rng = np.random.RandomState(42)
    initial_points = [x0_keich.copy()]

    # Add perturbed versions of Keich
    for scale in [0.01, 0.05, 0.1, 0.2, 0.5]:
        initial_points.append(x0_keich + rng.randn(n) * scale)

    # Add random points
    for _ in range(8):
        initial_points.append(rng.randn(n) * 0.5)

    # Add structured points based on different constructions
    # Reverse Keich
    initial_points.append(x0_keich[::-1].copy())

    # Shifted versions
    for shift in [0.1, -0.1, 0.2, -0.2]:
        initial_points.append(x0_keich + shift)

    # Sort initial points by their area (best first) to try good ones first
    initial_points.sort(key=lambda x: _union_area_and_grad(x)[0])

    # Try each initial point with L-BFGS-B
    for x_init in initial_points:
        if time.time() > deadline - 30:
            break

        x_try = _lbfgs_step(x_init, maxiter=300, gtol=1e-10)
        area_try = _union_area_and_grad(x_try)[0]

        if area_try < best_area:
            best_x = x_try.copy()
            best_area = area_try

        # If we found a significantly better solution, do a polish
        if area_try < best_area * 0.999:
            if time.time() > deadline - 60:
                break
            x_polished = _lbfgs_step(best_x, maxiter=500, gtol=1e-12)
            area_polished = _union_area_and_grad(x_polished)[0]
            if area_polished < best_area:
                best_x = x_polished.copy()
                best_area = area_polished

    # Final polish with tight tolerance
    if time.time() < deadline - 10:
        x_final = _lbfgs_step(best_x, maxiter=1000, gtol=1e-14)
        area_final = _union_area_and_grad(x_final)[0]
        if area_final < best_area:
            best_x = x_final.copy()
            best_area = area_final

    # Ensure valid output
    if not np.all(np.isfinite(best_x)):
        best_x = x0_keich

    return best_x


# EVOLVE-BLOCK-END
