# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _union_area(x, n=64, M=20000):
    """Compute the union area of n triangles given base positions x."""
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n
    L_sorted = np.sort(L, axis=1)
    diffs = np.diff(L_sorted, axis=1)
    union_len = w + np.sum(np.minimum(diffs, w[:, np.newaxis]), axis=1)
    return np.mean(union_len)

def _union_area_grad(x, n=64, M=20000):
    """Compute union area and its subgradient w.r.t. x."""
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n

    order = np.argsort(L, axis=1)  # (M, n)
    L_sorted = np.take_along_axis(L, order, axis=1)
    diffs = np.diff(L_sorted, axis=1)  # (M, n-1)

    active = (diffs < w[:, np.newaxis]).astype(np.float64)  # (M, n-1)
    union_len = w + np.sum(active * diffs, axis=1)
    area = np.mean(union_len)

    grad = np.zeros(n, dtype=np.float64)

    left_idx = order[:, :-1]   # (M, n-1)
    right_idx = order[:, 1:]   # (M, n-1)

    # For each active pair (m,j):
    #   grad[right_idx[m,j]] += active[m,j] * a[right_idx[m,j]] / M
    #   grad[left_idx[m,j]]  -= active[m,j] * a[left_idx[m,j]] / M

    a_left = a[left_idx]   # (M, n-1)
    a_right = a[right_idx]  # (M, n-1)

    np.add.at(grad, right_idx.ravel(), (active * a_right).ravel())
    np.add.at(grad, left_idx.ravel(), -(active * a_left).ravel())
    grad /= M

    return area, grad

def _keich_init(n=64):
    """Generate initial positions using Keich construction."""
    k = int(np.log2(n))
    j_arr = np.arange(1, k + 1)
    x0 = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0[i] = np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k)
    return x0

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    M = 20000
    deadline = time.time() + 580.0

    x0 = _keich_init(n)
    best_x = x0.copy()
    best_area = _union_area(x0, n, M)

    def objective_grad(x):
        area, grad = _union_area_grad(x, n, M)
        return area, grad

    # Phase 1: L-BFGS-B with exact gradient
    try:
        res = minimize(
            objective_grad,
            best_x,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500, 'gtol': 1e-8, 'maxfun': 500, 'disp': False}
        )
        if res.fun < best_area:
            best_x = res.x.copy()
            best_area = res.fun
    except Exception:
        pass

    # Phase 2: Nelder-Mead polish (handles non-smoothness)
    if time.time() < deadline - 30:
        try:
            res_nm = minimize(
                lambda x: _union_area(x, n, M),
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res_nm.fun < best_area:
                best_x = res_nm.x.copy()
                best_area = res_nm.fun
        except Exception:
            pass

    # Phase 3: Random restarts with L-BFGS-B
    np.random.seed(12345)
    for restart in range(15):
        if time.time() > deadline - 20:
            break

        noise_scale = 0.3
        x_start = best_x + noise_scale * np.random.randn(n)

        try:
            res = minimize(
                objective_grad,
                x_start,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'gtol': 1e-7, 'maxfun': 200, 'disp': False}
            )
            if res.fun < best_area:
                best_x = res.x.copy()
                best_area = res.fun

                # Polish with Nelder-Mead if time permits
                if time.time() < deadline - 20:
                    try:
                        res_nm = minimize(
                            lambda x: _union_area(x, n, M),
                            best_x,
                            method='Nelder-Mead',
                            options={'maxiter': 150, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
                        )
                        if res_nm.fun < best_area:
                            best_x = res_nm.x.copy()
                            best_area = res_nm.fun
                    except Exception:
                        pass
        except Exception:
            pass

    # Final validation
    if not np.all(np.isfinite(best_x)):
        return x0
    final_area = _union_area(best_x, n, M)
    if final_area <= 0:
        return x0

    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
