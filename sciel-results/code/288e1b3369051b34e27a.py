import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    t_start = time.time()
    deadline = 205.0  # 5s safety margin from 210s

    # Hermite polynomial values at 0 for the constraint P(0)=0
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0

    # Coefficients for H0, H4, H8, H12 as arrays indexed by power//2
    H0_arr = np.array([1.0])
    H4_arr = np.array([12.0, -48.0, 16.0])
    H8_arr = np.array([1680.0, -32256.0, 23040.0, -4608.0, 256.0])
    H12_arr = np.array([3991680.0, -12355200.0, 13271040.0, -5806080.0,
                        1032192.0, -98304.0, 4096.0])

    # Precompute ∫ x^{2k} e^{-πx²} dx for k=0..12
    E_int = np.zeros(13)
    E_int[0] = 1.0
    for k in range(1, 13):
        df = 1.0
        for j in range(1, 2 * k, 2):
            df *= j
        E_int[k] = df / (2.0 * np.pi) ** k

    # Signs for Q: Q[m] gets coefficient from P[2m] * (-1)^m
    signs = np.array([(-1.0) ** m for m in range(7)])

    def get_P(params):
        a0, a4, a8 = params
        a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
        
        P = np.zeros(7)
        P[0] += a0 * H0_arr[0]
        for m in range(3):
            P[m] += a4 * H4_arr[m]
        for m in range(5):
            P[m] += a8 * H8_arr[m]
        for m in range(7):
            P[m] += a12 * H12_arr[m]
        return P

    def objective_product(params):
        P = get_P(params)
        P2 = np.convolve(P, P)
        norm_sq = np.dot(P2, E_int)
        
        Q = P * signs
        Q2 = np.convolve(Q, Q)
        norm_sq_hat = np.dot(Q2, E_int)
        
        if norm_sq <= 0 or norm_sq_hat <= 0:
            return 1e30
        return norm_sq * norm_sq_hat

    def objective_log(params):
        P = get_P(params)
        P2 = np.convolve(P, P)
        norm_sq = np.dot(P2, E_int)
        
        Q = P * signs
        Q2 = np.convolve(Q, Q)
        norm_sq_hat = np.dot(Q2, E_int)
        
        if norm_sq <= 0 or norm_sq_hat <= 0:
            return 1e30
        return np.log(norm_sq) + np.log(norm_sq_hat)

    # Initial guess from the seed program
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_result = None
    best_score = np.inf

    # 1. Global Search using Differential Evolution
    bounds = [(-10.0, 10.0), (-1.0, 1.0), (-0.1, 0.1)]
    
    if time.time() < deadline - 60:
        try:
            de_result = differential_evolution(
                objective_product, 
                bounds, 
                seed=42, 
                maxiter=2000, 
                tol=1e-12,
                polish=True,
                update_interval=100,
                workers=1,
                popsize=30
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_result = de_result.x.copy()
        except Exception as e:
            pass

    # 2. Local Search from multiple starting points using Nelder-Mead
    start_points = [
        x0,
        best_result if best_result is not None else x0,
        np.array([-1.0, 0.1, 0.01]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-0.5, 0.05, 0.005]),
        np.array([-1.2, 0.05, 0.001]),
        np.array([-0.8, 0.04, 0.002]),
        np.array([-1.1, 0.08, 0.003]),
        np.array([-0.9, 0.06, 0.004]),
        np.array([-1.3, 0.06, 0.002]),
        np.array([-1.05, 0.045, 0.0003]),
        np.array([-1.15, 0.035, 0.0004]),
    ]
    
    for sp in start_points:
        if time.time() > deadline - 30:
            break
        if sp is None:
            continue
        try:
            result = minimize(
                objective_product,
                sp,
                method='Nelder-Mead',
                options={'maxiter': 50000, 'xatol': 1e-15, 'fatol': 1e-17,
                         'adaptive': True}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            continue

    # 3. High-Precision Polish with BFGS on the log-objective
    if best_result is not None and time.time() < deadline - 10:
        try:
            result = minimize(
                objective_log,
                best_result,
                method='BFGS',
                options={'maxiter': 20000, 'gtol': 1e-15, 'disp': False}
            )
            new_score = objective_product(result.x)
            if new_score < best_score:
                best_score = new_score
                best_result = result.x.copy()
        except Exception as e:
            pass

    # 4. Final ultra-precise Nelder-Mead polish on the product objective
    if best_result is not None and time.time() < deadline - 5:
        try:
            result = minimize(
                objective_product,
                best_result,
                method='Nelder-Mead',
                options={'maxiter': 100000, 'xatol': 1e-16, 'fatol': 1e-18,
                         'adaptive': True}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            pass

    # 5. Extra polish with L-BFGS-B on the log objective if time permits
    if best_result is not None and time.time() < deadline - 2:
        try:
            result = minimize(
                objective_log,
                best_result,
                method='L-BFGS-B',
                options={'maxiter': 50000, 'gtol': 1e-15, 'ftol': 1e-15, 'disp': False}
            )
            new_score = objective_product(result.x)
            if new_score < best_score:
                best_score = new_score
                best_result = result.x.copy()
        except Exception:
            pass

    if best_result is None:
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    return best_result
