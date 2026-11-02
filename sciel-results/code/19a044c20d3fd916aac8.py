"""Rewritten program: fast numerical optimization of Hermite coefficients for uncertainty bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from numpy.polynomial.hermite import hermgauss


def construct():
    """Return three coefficients for H0, H4, H8 Hermite combination."""
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin for 210s wall-clock limit

    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    # Gauss-Hermite quadrature nodes and weights
    n_quad = 200
    t_nodes, t_weights = hermgauss(n_quad)
    x_nodes = np.sqrt(2.0) * t_nodes
    x_weights = np.sqrt(2.0) * t_weights

    # Precompute Hermite polynomials at quadrature nodes
    x2 = x_nodes**2
    x4 = x2**2
    x6 = x4 * x2
    x8 = x4**2
    x10 = x8 * x2
    x12 = x6**2

    H0 = np.ones_like(x_nodes)
    H4 = 16.0 * x4 - 48.0 * x2 + 12.0
    H8 = 256.0 * x8 - 2304.0 * x6 + 5760.0 * x4 - 4320.0 * x2 + 504.0
    H12 = (4096.0 * x12 - 55296.0 * x10 + 303104.0 * x8
           - 589824.0 * x6 + 408192.0 * x4 - 77760.0 * x2 + 10395.0)

    # Precompute weighted powers for fast norm computation
    # We need: sum(w * P^k) for k=2,4 where P = a0*H0 + a4*H4 + a8*H8 + a12*H12
    # and FP = a0*H0 + a4*H4 + a8*H8 - a12*H12

    def get_a12(a0, a4, a8):
        """Compute a12 so that g(0) = 0."""
        return (a0 - 3.0 * a4 + 105.0 * a8) / 10395.0

    def compute_P(a0, a4, a8, a12):
        """Compute P and FP at quadrature nodes."""
        P = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
        FP = a0 * H0 + a4 * H4 + a8 * H8 - a12 * H12
        return P, FP

    def objective(coeffs):
        """Compute the upper bound on C4 for the given Hermite coefficients."""
        a0, a4, a8 = coeffs
        a12 = get_a12(a0, a4, a8)

        P, FP = compute_P(a0, a4, a8, a12)

        # L2 norms
        norm2_g_sq = np.sum(x_weights * P**2)
        norm2_fg_sq = np.sum(x_weights * FP**2)

        if norm2_g_sq < 1e-15 or norm2_fg_sq < 1e-15:
            return 1e10

        norm2_g = np.sqrt(norm2_g_sq)
        norm2_fg = np.sqrt(norm2_fg_sq)

        # L4 norms
        norm4_g_4 = np.sum(x_weights * P**4)
        norm4_fg_4 = np.sum(x_weights * FP**4)

        if norm4_g_4 < 1e-15 or norm4_fg_4 < 1e-15:
            return 1e10

        norm4_g = norm4_g_4 ** 0.25
        norm4_fg = norm4_fg_4 ** 0.25

        # The bound on C4
        bound = (norm4_g * norm4_fg) / (norm2_g * norm2_fg)

        return bound

    best_coeffs = seed.copy()
    best_val = objective(seed)

    # Global optimization with differential evolution - more aggressive
    if time.time() < deadline - 60.0:
        try:
            bounds = [(-5.0, 5.0), (-0.5, 0.5), (-0.01, 0.01)]
            de_result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=200,
                tol=1e-12,
                popsize=20,
                mutation=(0.3, 1.2),
                recombination=0.7,
                polish=False,
                init='sobol'
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_coeffs = de_result.x.copy()
        except Exception:
            pass

    # Local refinement from best point using L-BFGS-B
    if time.time() < deadline - 30.0:
        try:
            result = minimize(
                objective,
                best_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    # Nelder-Mead refinement
    if time.time() < deadline - 15.0:
        try:
            result = minimize(
                objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    # Additional local refinement from seed
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                objective,
                seed,
                method='L-BFGS-B',
                options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    # Final tight refinement
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-16, 'fatol': 1e-18, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
