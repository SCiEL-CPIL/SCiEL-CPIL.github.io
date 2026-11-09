# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _union_area_and_grad(x, M=20000):
    """Compute the union-area objective and its gradient.

    x: (64,) base positions.
    Returns (area, grad).
    """
    n = 64
    w0 = 1.0 / n
    a = (np.arange(n) + 1) / n  # (64,)
    y = (np.arange(M) + 0.5) / M  # (M,)

    # L[m, i] = x_i + a_i * y_m   shape (M, n)
    L = x[None, :] + a[None, :] * y[:, None]

    # Sort L along axis 1
    Ls = np.sort(L, axis=1)
    # Track which original index ended up in each sorted slot
    idx = np.argsort(L, axis=1)  # (M, n), idx[m, s] = original i

    w = w0 * (1.0 - y)  # (M,)

    # Consecutive gaps in sorted order
    gaps = Ls[:, 1:] - Ls[:, :-1]  # (M, n-1)
    # Effective contribution of each gap: min(gap, w)
    eff = np.minimum(gaps, w[:, None])  # (M, n-1)

    # Union length at each height
    union_len = w + eff.sum(axis=1)  # (M,)
    area = union_len.mean()

    # Gradient: d(area)/d(x_i)
    # eff[m, s] depends on Ls[m, s+1] - Ls[m, s]
    # d(eff[m,s])/d(Ls[m,s])  = -1 if gaps[m,s] < w_m else 0
    # d(eff[m,s])/d(Ls[m,s+1]) = +1 if gaps[m,s] < w_m else 0
    # (at gap == w it's a kink; we pick subgradient 0)
    active = gaps < w[:, None]  # (M, n-1)

    # d(eff)/d(Ls): shape (M, n)
    d_eff_d_Ls = np.zeros_like(Ls)
    # For slot s (0-based) in Ls:
    #   from eff[m, s] (gap between slot s and s+1): contributes -active[m,s] to slot s
    #   from eff[m, s-1] (gap between slot s-1 and s): contributes +active[m,s-1] to slot s
    d_eff_d_Ls[:, 0] = -active[:, 0]
    d_eff_d_Ls[:, -1] = active[:, -2]
    d_eff_d_Ls[:, 1:-1] = active[:, :-1] - active[:, 1:]

    # d(union_len[m])/d(Ls[m, s]) = d_eff_d_Ls[m, s]
    # d(area)/d(x_i) = (1/M) * sum_m d(union_len[m])/d(L[m, i])
    # d(union_len[m])/d(L[m, i]) = d_eff_d_Ls[m, idx[m, i]]
    # Gather: for each m, scatter d_eff_d_Ls[m, :] back to original indices
    # grad[i] = (1/M) * sum_m d_eff_d_Ls[m, idx[m, i]]
    # Equivalent: grad[i] = (1/M) * sum_m sum_s d_eff_d_Ls[m, s] * 1{idx[m,s] == i}
    # Use scatter-add
    grad = np.zeros(n)
    # For each m, add d_eff_d_Ls[m, s] to grad[idx[m, s]]
    # Vectorized: for each slot s, accumulate over m
    for s in range(n):
        grad += d_eff_d_Ls[:, s][idx[:, s]]
    grad /= M
    return area, grad


def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)

    def keich():
        positions = []
        for i in range(n):
            eps = np.array(list(str(bin(i))[2:]), dtype=int)
            eps = np.pad(eps, (k - len(eps), 0), "constant")
            positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
        return np.array(positions, dtype=np.float64)

    t_start = time.time()
    deadline = 620.0  # safety margin before 630s kill

    # Baseline (Keich)
    x_keich = keich()
    area_keich, _ = _union_area_and_grad(x_keich, M=2000)
    best_x = x_keich.copy()
    best_area = area_keich

    rng = np.random.default_rng(42)

    # Multi-restart L-BFGS-B with moderate M for speed, then final polish at full M
    restarts = 20
    for r in range(restarts):
        if time.time() - t_start > deadline - 60.0:
            break
        # Diverse initial points
        if r == 0:
            x0 = x_keich.copy()
        else:
            # Perturb keich or use random
            if r % 2 == 1:
                x0 = x_keich + rng.normal(0, 0.05, n)
            else:
                x0 = rng.uniform(-0.5, 1.5, n)

        M_opt = 5000 if r < 10 else 10000

        def obj_and_grad(xx):
            return _union_area_and_grad(xx, M=M_opt)

        res = minimize(
            obj_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50},
        )
        if res.success and res.fun < best_area:
            best_area = res.fun
            best_x = res.x.copy()

    # Final polish at full resolution M=20000
    if time.time() - t_start < deadline - 30.0:
        res = minimize(
            lambda xx: _union_area_and_grad(xx, M=20000),
            best_x,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100},
        )
        if res.success and res.fun < best_area:
            best_area = res.fun
            best_x = res.x.copy()

    # Safety: ensure finite
    if not np.all(np.isfinite(best_x)):
        best_x = x_keich

    return best_x.astype(np.float64)


# EVOLVE-BLOCK-END
