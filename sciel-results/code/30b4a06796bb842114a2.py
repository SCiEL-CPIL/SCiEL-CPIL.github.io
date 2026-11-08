import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 580

    m = 12

    # Precompute the transformation matrix from monomial basis to Laguerre basis
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    A = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k + 1, j + 1) / math.factorial(j)

    d = np.arange(m + 1) + 1

    try:
        A_inv = np.linalg.inv(A)
        K = A_inv @ np.diag(d) @ A_inv.T
    except np.linalg.LinAlgError:
        A_pinv = np.linalg.pinv(A)
        K = A_pinv @ np.diag(d) @ A_pinv.T

    def compute_norm_sq(roots):
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]
        norm_sq = float(p @ K @ p)
        if norm_sq < 0:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        roots = np.exp(log_roots)
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
        norm_sq = compute_norm_sq(roots)
        if norm_sq <= 1e-10:
            return 1e10
        return 1.0 / norm_sq

    seed_log_roots = np.log(4.0) + np.arange(12) * np.log(2.0)
    bounds = [(-4.0, 10.0)] * 12

    best_roots = None
    best_val = 1e10
    best_log_roots = None

    seed_val = cost_func(seed_log_roots)
    best_val = seed_val
    best_roots = np.exp(seed_log_roots)
    best_log_roots = seed_log_roots.copy()

    r1 = 1.0 * (10.0 ** (np.arange(12) / 11.0))
    seeds = [seed_log_roots, np.log(r1)]

    lag_coeffs_desc = np.zeros(13)
    for k in range(13):
        if k <= 12:
            lag_coeffs_desc[12 - k] = ((-1)**k) * math.comb(13, 12 - k) / math.factorial(k)

    try:
        lag_roots = np.roots(lag_coeffs_desc)
        lag_roots_real = lag_roots[np.abs(lag_roots.imag) < 1e-9].real
        lag_roots_pos = lag_roots_real[lag_roots_real > 0]
        if len(lag_roots_pos) == 12:
            seeds.append(np.log(lag_roots_pos))
    except Exception:
        pass

    for seed_log in seeds:
        if time.time() > deadline - 100:
            break
        try:
            res = minimize(
                cost_func, seed_log, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-14}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_roots = np.exp(res.x)
                best_log_roots = res.x.copy()
        except Exception:
            pass

    de_runs = 0
    remaining_time = deadline - time.time()
    max_de_runs = 2
    for i in range(max_de_runs):
        if time.time() > deadline - 120:
            break
        de_seed = 123 + i * 100
        try:
            result = differential_evolution(
                cost_func, bounds, seed=de_seed, maxiter=100,
                popsize=15, tol=1e-8, polish=False, init='sobol'
            )
            if result.fun < best_val:
                best_val = result.fun
                best_roots = np.exp(result.x)
                best_log_roots = result.x.copy()
            de_runs += 1
        except Exception:
            pass

    if best_log_roots is not None and time.time() < deadline - 10:
        try:
            polish_result = minimize(
                cost_func, best_log_roots, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_roots = np.exp(polish_result.x)
                best_log_roots = polish_result.x.copy()
        except Exception:
            pass

    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)

    return [4.0 * (2.0 ** i) for i in range(12)]
