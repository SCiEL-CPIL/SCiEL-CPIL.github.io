"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hermite_physicists(n, x):
    """Physicists' Hermite polynomials H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    H_prev = np.ones_like(x)
    H_curr = 2 * x
    for k in range(1, n):
        H_next = 2 * x * H_curr - 2 * k * H_prev
        H_prev = H_curr
        H_curr = H_next
    return H_curr


def _compute_bound(a0, a4, a8, n_points=20000, x_range=15.0):
    """Compute the uncertainty product bound for given coefficients."""
    # H12 coefficient from g(0) = 0
    # H_n(0) for even n: H_0(0)=1, H_4(0)=12, H_8(0)=168, H_12(0)=3024
    a12 = -(a0 + a4 * 12.0 + a8 * 168.0) / 3024.0

    x = np.linspace(-x_range, x_range, n_points)
    dx = x[1] - x[0]

    H0 = np.ones_like(x)
    H4 = 16 * x**4 - 48 * x**2 + 12
    H8 = 256 * x**8 - 1344 * x**6 + 2688 * x**4 - 1344 * x**2 + 168
    H12 = _hermite_physicists(12, x)

    gauss = np.exp(-x**2 / 2.0)

    g = (a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12) * gauss

    # Fourier transform: for H_n(x)*e^{-x^2/2}, FT is (-i)^n H_n(k)*e^{-k^2/2}
    # For even n, (-i)^n = (-1)^{n/2}
    k = x  # same grid
    dk = dx

    Fg = (a0 * H0 - a4 * H4 + a8 * H8 - a12 * H12) * gauss

    # L1 norms
    norm_g = np.sum(np.abs(g)) * dx
    norm_Fg = np.sum(np.abs(Fg)) * dk

    # Handle near-zero norms
    if norm_g < 1e-15 or norm_Fg < 1e-15:
        return 1e10

    return norm_g * norm_Fg


def _objective(coeffs):
    """Objective function for optimization."""
    a0, a4, a8 = coeffs
    return _compute_bound(a0, a4, a8)


def construct():
    start_time = time.time()
    deadline = start_time + 190.0  # 20s safety margin before 210s kill

    # Seed values
    seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_score = _compute_bound(seed[0], seed[1], seed[2])
    best_coeffs = seed.copy()

    # Phase 1: Grid search for good starting points
    if time.time() < deadline - 60.0:
        # Coarse grid
        a0_range = np.linspace(-3.0, 3.0, 15)
        a4_range = np.linspace(-0.5, 0.5, 10)
        a8_range = np.linspace(-0.02, 0.02, 8)

        best_grid_score = best_score
        best_grid_coeffs = best_coeffs.copy()

        for a0 in a0_range:
            for a4 in a4_range:
                for a8 in a8_range:
                    if time.time() > deadline - 60.0:
                        break
                    score = _compute_bound(a0, a4, a8, n_points=5000, x_range=12.0)
                    if score < best_grid_score:
                        best_grid_score = score
                        best_grid_coeffs = np.array([a0, a4, a8])

        if best_grid_score < best_score:
            best_score = best_grid_score
            best_coeffs = best_grid_coeffs

    # Phase 2: Nelder-Mead refinement from multiple starting points
    starting_points = [best_coeffs.copy(), seed.copy()]

    # Add some diverse starting points
    for _ in range(5):
        if time.time() > deadline - 45.0:
            break
        sp = best_coeffs + 0.1 * np.random.randn(3)
        starting_points.append(sp)

    for sp in starting_points:
        if time.time() > deadline - 30.0:
            break
        try:
            result = minimize(
                _objective,
                sp,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            if result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            continue

    # Phase 3: High-precision polish with tighter grid and more points
    if time.time() < deadline - 10.0:
        # Fine grid around best solution
        center = best_coeffs
        fine_a0 = np.linspace(center[0] - 0.05, center[0] + 0.05, 11)
        fine_a4 = np.linspace(center[1] - 0.02, center[1] + 0.02, 9)
        fine_a8 = np.linspace(center[2] - 0.005, center[2] + 0.005, 7)

        for a0 in fine_a0:
            for a4 in fine_a4:
                for a8 in fine_a8:
                    if time.time() > deadline - 10.0:
                        break
                    score = _compute_bound(a0, a4, a8, n_points=10000, x_range=14.0)
                    if score < best_score:
                        best_score = score
                        best_coeffs = np.array([a0, a4, a8])

        # Final Nelder-Mead polish
        if time.time() < deadline - 5.0:
            try:
                result = minimize(
                    _objective,
                    best_coeffs,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 1000,
                        'xatol': 1e-12,
                        'fatol': 1e-14,
                        'adaptive': True
                    }
                )
                if result.fun < best_score:
                    best_score = result.fun
                    best_coeffs = result.x.copy()
            except Exception:
                pass

    # Final verification with high precision
    if time.time() < deadline - 2.0:
        final_score = _compute_bound(best_coeffs[0], best_coeffs[1], best_coeffs[2],
                                     n_points=50000, x_range=20.0)
        if final_score < best_score:
            best_score = final_score

    return best_coeffs
# EVOLVE-BLOCK-END
