import numpy as np
from scipy.optimize import minimize
import time


def construct():
    t_start = time.time()
    deadline = 198.0  # safety margin from the 210 s wall-clock limit

    # ---- exact Gaussian-weighted moment integrals ----
    # I_k = ∫ x^{2k} e^{-πx²} dx = (2k-1)!! / (2π)^k   (I_0 = 1)
    E_int = np.zeros(13)
    E_int[0] = 1.0
    for k in range(1, 13):
        df = 1.0
        for j in range(1, 2 * k, 2):
            df *= j
        E_int[k] = df / (2.0 * np.pi) ** k

    # Hermite polynomials (physicists'), coefficients over even powers x^0..x^{2m}
    H0_arr = np.array([1.0])
    H4_arr = np.array([12.0, -48.0, 16.0])
    H8_arr = np.array([1680.0, -32256.0, 23040.0, -4608.0, 256.0])
    H12_arr = np.array([3991680.0, -12355200.0, 13271040.0, -5806080.0,
                        1032192.0, -98304.0, 4096.0])

    # values at 0 for the P(0)=0 constraint
    H0_0, H4_0, H8_0, H12_0 = 1.0, 12.0, 1680.0, 665280.0

    # P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12,  a12 = -(a0+12 a4+1680 a8)/665280
    # Build the 7x3 matrix M with P[m] = sum_j M[m,j] * a_j  (j -> a0,a4,a8)
    M = np.zeros((7, 3))
    for m in range(1):
        M[m, 0] += H0_arr[m]
    for m in range(3):
        M[m, 1] += H4_arr[m]
    for m in range(5):
        M[m, 2] += H8_arr[m]
    for m in range(7):
        M[m, 2] += (H12_arr[m] * (-H12_arr[m] / H12_0)) * 0.0  # placeholder (kept zero)
    # H12 contribution: a12 = -(a0*1 + a4*12 + a8*1680)/665280
    h12_row = H12_arr / H12_0
    for j, hj in enumerate([1.0, H4_0, H8_0]):
        M[:, j] += -hj * h12_row

    # Gram matrix G[i,j] = I_{i+j}
    G = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            G[i, j] = E_int[i + j]

    # Fourier side: Q[m] = (-1)^m P[m]  ->  ||ĝ||² = P^T (S G S) P, S=diag((-1)^m)
    s = np.array([(-1.0) ** m for m in range(7)])
    H = (s[:, None] * G) * s[None, :]

    # objective = (a^T G a) * (a^T H a)
    def objective(a):
        ga = a @ G @ a
        ha = a @ H @ a
        if ga <= 0.0 or ha <= 0.0:
            return 1e30
        return ga * ha

    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best = seed.copy()
    best_val = objective(seed)

    def try_optimize(x0, method, options, budget_left):
        nonlocal best, best_val
        if time.time() > deadline - budget_left:
            return
        try:
            res = minimize(objective, x0, method=method, options=options)
            if res is not None and np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best = np.asarray(res.x, dtype=float).copy()
        except Exception:
            pass

    # ---- 1) coarse global grid (cheap, gives many diverse basins) ----
    grid = []
    for a0 in np.linspace(-2.0, 0.0, 15):
        for a4 in np.linspace(-0.4, 0.4, 9):
            for a8 in np.linspace(-0.05, 0.05, 5):
                grid.append(np.array([a0, a4, a8]))
    for g in grid:
        v = objective(g)
        if v < best_val:
            best_val = v
            best = g.copy()

    # ---- 2) local polish from the grid minima + seed + a few hand points ----
    nm_opts = {'maxiter': 30000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
    # polish the best grid point first, then a spread of the top grid points
    grid_vals = [(objective(g), g) for g in grid]
    grid_vals.sort(key=lambda t: t[0])
    candidates = [seed] + [g for _, g in grid_vals[:24]] + [
        np.array([-1.0, 0.1, 0.01]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-0.5, 0.05, 0.005]),
        np.array([-1.2, 0.05, 0.001]),
        np.array([-0.8, 0.04, 0.002]),
    ]
    for sp in candidates:
        if time.time() > deadline - 20:
            break
        try_optimize(sp, 'Nelder-Mead', nm_opts, 5)

    # ---- 3) gradient-based polish (BFGS) from the current best ----
    if time.time() < deadline - 10:
        try_optimize(best, 'BFGS',
                     {'maxiter': 20000, 'gtol': 1e-16, 'disp': False}, 5)
        # a few BFGS starts from other good points
        for sp in [g for _, g in grid_vals[:6]]:
            if time.time() > deadline - 6:
                break
            try_optimize(sp, 'BFGS',
                         {'maxiter': 20000, 'gtol': 1e-16, 'disp': False}, 3)

    # ---- 4) final ultra-tight Nelder-Mead on the best ----
    if time.time() < deadline - 2:
        try_optimize(best, 'Nelder-Mead',
                     {'maxiter': 60000, 'xatol': 1e-16, 'fatol': 1e-18,
                      'adaptive': True}, 1)

    if not np.all(np.isfinite(best)):
        return seed
    return best
