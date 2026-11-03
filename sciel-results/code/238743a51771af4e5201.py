import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


def construct():
    t_start = time.time()
    deadline = 205.0  # 5s safety margin from 210s

    # Hermite polynomial values at 0 for the constraint P(0) = 0
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0

    # Coefficients for powers x^0, x^2, ..., x^12 (index m -> power 2m)
    H0_arr = np.array([1.0])
    H4_arr = np.array([12.0, -48.0, 16.0])
    H8_arr = np.array([1680.0, -32256.0, 23040.0, -4608.0, 256.0])
    H12_arr = np.array([3991680.0, -12355200.0, 13271040.0, -5806080.0,
                        1032192.0, -98304.0, 4096.0])

    # Precompute ∫ x^{2k} e^{-πx²} dx for k=0..12  = (2k-1)!! / (2π)^k
    E_int = np.zeros(13)
    E_int[0] = 1.0
    for k in range(1, 13):
        df = 1.0
        for j in range(1, 2 * k, 2):
            df *= j
        E_int[k] = df / (2.0 * np.pi) ** k

    # Signs for Q: Q[m] = P[m] * (-1)^m
    signs = np.array([(-1.0) ** m for m in range(7)])

    # ---- Spherical reparameterization over the P(0)=0 nullspace ----
    # Constraint: a0 + 12 a4 + 1680 a8 + 665280 a12 = 0
    # Nullspace basis (a0, a4, a8, a12):
    v1 = np.array([0.0, 1.0, 0.0, -12.0 / 665280.0])
    v2 = np.array([0.0, 0.0, 1.0, -1680.0 / 665280.0])
    # Orthonormalize in R^4
    u1 = v1 / np.linalg.norm(v1)
    u2 = v2 - np.dot(v2, u1) * u1
    u2 = u2 / np.linalg.norm(u2)
    U = np.vstack([u1, u2])  # 2x4

    # Seed point for reference (lies in nullspace up to rounding)
    a_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    a12_seed = -(a_seed[0] * H0_0 + a_seed[1] * H4_0 + a_seed[2] * H8_0) / H12_0
    a_seed_full = np.array([a_seed[0], a_seed[1], a_seed[2], a12_seed])
    seed_r = np.dot(U, a_seed_full)  # [r1, r2]
    seed_norm = np.linalg.norm(seed_r)
    seed_theta = np.arctan2(seed_r[1], seed_r[0])

    def build_P(a):
        a0, a4, a8, a12 = a
        P = np.zeros(7)
        P[0] += a0 * H0_arr[0]
        for m in range(3):
            P[m] += a4 * H4_arr[m]
        for m in range(5):
            P[m] += a8 * H8_arr[m]
        for m in range(7):
            P[m] += a12 * H12_arr[m]
        return P

    def objective_from_a(a):
        P = build_P(a)
        P2 = np.convolve(P, P)
        norm_sq = np.dot(P2, E_int)
        Q = P * signs
        Q2 = np.convolve(Q, Q)
        norm_sq_hat = np.dot(Q2, E_int)
        return norm_sq * norm_sq_hat

    def objective_spherical(params):
        s, theta = params
        a = s * (np.cos(theta) * U[0] + np.sin(theta) * U[1])
        return objective_from_a(a)

    def objective_log(params):
        val = objective_from_a(params)
        if val <= 0:
            return 0.0
        return np.log(val)

    best_a = a_seed_full.copy()
    best_val = objective_from_a(a_seed_full)

    def consider(a, val):
        nonlocal best_a, best_val
        if np.isfinite(val) and val < best_val:
            best_val = val
            best_a = a.copy()

    # ---- Global search in (s, theta) via Differential Evolution ----
    if time.time() < deadline - 70:
        try:
            s_lo, s_hi = 0.0, 5.0
            bounds = [(s_lo, s_hi), (-np.pi, np.pi)]
            res = differential_evolution(
                objective_spherical,
                bounds=bounds,
                seed=12345,
                maxiter=120,
                tol=1e-10,
                popsize=18,
                polish=True,
                init='sobol',
                updating='deferred',
            )
            if res.fun < best_val:
                best_val = res.fun
                best_a = (res.x[0] * (np.cos(res.x[1]) * U[0]
                                       + np.sin(res.x[1]) * U[1])).copy()
        except Exception:
            pass

    # ---- Local polish in 3D (a0, a4, a8) with exact a12 ----
    a3 = best_a[:3].copy()

    def try_local(a3_start, method, opts, tag):
        nonlocal best_a, best_val
        if time.time() > deadline - 10:
            return
        try:
            r = minimize(objective_from_a_3, a3_start, method=method, options=opts)
            if r.fun < best_val:
                best_val = r.fun
                best_a = np.array([r.x[0], r.x[1], r.x[2],
                                   -(r.x[0] * H0_0 + r.x[1] * H4_0 + r.x[2] * H8_0) / H12_0])
        except Exception:
            pass

    def objective_from_a_3(a3):
        a = np.array([a3[0], a3[1], a3[2],
                      -(a3[0] * H0_0 + a3[1] * H4_0 + a3[2] * H8_0) / H12_0])
        return objective_from_a(a)

    # Nelder-Mead polish
    try_local(a3, 'Nelder-Mead',
              {'maxiter': 40000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True}, 'nm1')
    # BFGS polish
    try_local(a3, 'BFGS', {'maxiter': 20000, 'gtol': 1e-18, 'disp': False}, 'bfgs')
    # L-BFGS-B polish
    try_local(a3, 'L-BFGS-B', {'maxiter': 20000, 'ftol': 1e-18, 'gtol': 1e-18}, 'lbfgs')
    # Powell polish
    try_local(a3, 'Powell', {'maxiter': 20000, 'xtol': 1e-15, 'ftol': 1e-18}, 'powell')

    # ---- Final high-precision Nelder-Mead on best point ----
    if time.time() < deadline - 5:
        try:
            a3_final = best_a[:3].copy()
            r = minimize(objective_from_a_3, a3_final, method='Nelder-Mead',
                         options={'maxiter': 120000, 'xatol': 1e-18, 'fatol': 1e-20,
                                  'adaptive': True})
            if r.fun < best_val:
                best_val = r.fun
                best_a = np.array([r.x[0], r.x[1], r.x[2],
                                   -(r.x[0] * H0_0 + r.x[1] * H4_0 + r.x[2] * H8_0) / H12_0])
        except Exception:
            pass

    # Fallback to seed if anything went wrong
    if not np.all(np.isfinite(best_a)) or best_val > objective_from_a(a_seed_full):
        best_a = a_seed_full

    return np.array([best_a[0], best_a[1], best_a[2]])
