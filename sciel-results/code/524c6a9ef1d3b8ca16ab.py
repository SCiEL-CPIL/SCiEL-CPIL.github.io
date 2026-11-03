import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


def construct():
    t_start = time.time()
    deadline = 205.0  # 5s safety margin from 210s

    # Hermite polynomial values at 0 for the constraint
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

    def objective(params):
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

        P2 = np.convolve(P, P)
        norm_sq = np.dot(P2, E_int)

        Q = P * signs
        Q2 = np.convolve(Q, Q)
        norm_sq_hat = np.dot(Q2, E_int)

        return norm_sq * norm_sq_hat

    # Initial guess from the seed program
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    best_result = None
    best_score = np.inf

    # Strategy 1: Differential Evolution over 3 parameters with broader bounds
    if time.time() < deadline - 80:
        try:
            bounds = [(-3.0, 0.0), (-0.5, 0.5), (-0.05, 0.05)]
            de_result = differential_evolution(
                objective,
                bounds=bounds,
                seed=42,
                maxiter=500,
                tol=1e-14,
                atol=1e-16,
                popsize=50,
                polish=False,
                init='latinhypercube',
                mutation=(0.5, 1.0),
                recombination=0.7
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_result = de_result.x.copy()
        except Exception:
            pass

    # Strategy 2: Local optimization from multiple diverse starting points
    start_points = [x0]

    if best_result is not None:
        start_points.append(best_result)
        # Add perturbations around the DE result
        for scale in [0.001, 0.01, 0.1, 0.5]:
            for dim in range(3):
                perturbed = best_result.copy()
                perturbed[dim] += scale
                start_points.append(perturbed.copy())
                perturbed[dim] -= scale
                start_points.append(perturbed.copy())

    # Add more diverse starting points
    seed_perturbations = [
        np.array([-1.13, 0.04, 0.0003]),
        np.array([-1.10, 0.04, 0.0003]),
        np.array([-1.16, 0.04, 0.0003]),
        np.array([-1.13, 0.03, 0.0003]),
        np.array([-1.13, 0.05, 0.0003]),
        np.array([-1.13, 0.04, 0.0002]),
        np.array([-1.13, 0.04, 0.0004]),
        np.array([-1.13, 0.04, 0.00035]),
        np.array([-1.13, 0.04, 0.00025]),
        np.array([-1.12, 0.041, 0.00031]),
        np.array([-1.14, 0.039, 0.00029]),
        np.array([-1.11, 0.042, 0.00032]),
        np.array([-1.15, 0.038, 0.00028]),
        np.array([-1.20, 0.05, 0.0005]),
        np.array([-1.00, 0.03, 0.0002]),
        np.array([-1.30, 0.06, 0.0006]),
        np.array([-0.90, 0.02, 0.0001]),
    ]
    start_points.extend(seed_perturbations)

    for sp in start_points:
        if time.time() > deadline - 50:
            break

        try:
            # Use Nelder-Mead with adaptive simplex
            result = minimize(
                objective,
                sp,
                method='Nelder-Mead',
                options={'maxiter': 200000, 'xatol': 1e-18, 'fatol': 1e-20,
                         'adaptive': True}
            )

            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()

        except Exception:
            continue

    # Strategy 3: BFGS polish
    if best_result is not None and time.time() < deadline - 30:
        try:
            result = minimize(
                objective,
                best_result,
                method='BFGS',
                options={'maxiter': 200000, 'gtol': 1e-22, 'disp': False}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            pass

    # Strategy 4: L-BFGS-B polish
    if best_result is not None and time.time() < deadline - 20:
        try:
            result = minimize(
                objective,
                best_result,
                method='L-BFGS-B',
                options={'maxiter': 200000, 'ftol': 1e-22, 'gtol': 1e-22}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            pass

    # Strategy 5: Powell's method for final refinement
    if best_result is not None and time.time() < deadline - 10:
        try:
            result = minimize(
                objective,
                best_result,
                method='Powell',
                options={'maxiter': 200000, 'xtol': 1e-18, 'ftol': 1e-21}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            pass

    # Strategy 6: Final ultra-high-precision Nelder-Mead
    if best_result is not None and time.time() < deadline - 3:
        try:
            result = minimize(
                objective,
                best_result,
                method='Nelder-Mead',
                options={'maxiter': 500000, 'xatol': 1e-20, 'fatol': 1e-22,
                         'adaptive': True}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            pass

    # If no valid result, fall back to seed
    if best_result is None:
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    return best_result
