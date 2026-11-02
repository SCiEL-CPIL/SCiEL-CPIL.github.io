"""Rewritten program: differential evolution + L-BFGS-B optimization of Hermite coefficients."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize, differential_evolution


def construct():
    """Return three coefficients for H0, H4, H8 Hermite combination."""
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin for 210s wall-clock limit

    # Seed coefficients
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    # Gauss-Hermite quadrature with higher accuracy
    n_quad = 500
    t_nodes, t_weights = hermgauss(n_quad)
    x_nodes = np.sqrt(2.0) * t_nodes
    x_weights = np.sqrt(2.0) * t_weights

    # Precompute powers of x
    x2 = x_nodes**2
    x4 = x2**2
    x6 = x4 * x2
    x8 = x4**2
    x10 = x8 * x2
    x12 = x6**2

    # Precompute Hermite polynomials at quadrature nodes
    H0 = np.ones_like(x_nodes)
    H4 = 16.0 * x4 - 48.0 * x2 + 12.0
    H8 = 256.0 * x8 - 2304.0 * x6 + 5760.0 * x4 - 4320.0 * x2 + 504.0
    H12 = (4096.0 * x12 - 55296.0 * x10 + 303104.0 * x8
           - 589824.0 * x6 + 408192.0 * x4 - 77760.0 * x2 + 10395.0)

    def get_a12(a0, a4, a8):
        """Compute a12 so that g(0) = 0."""
        return (a0 - 3.0 * a4 + 105.0 * a8) / 10395.0

    def compute_bound_and_gradient(coeffs):
        """Compute the C4 upper bound and its gradient analytically."""
        a0, a4, a8 = coeffs
        a12 = get_a12(a0, a4, a8)

        # P(x) and F(P)(x) at quadrature nodes
        P = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
        FP = a0 * H0 + a4 * H4 + a8 * H8 - a12 * H12  # i^12 = -1

        # L2 norms
        norm2_g_sq = np.sum(x_weights * P**2)
        norm2_fg_sq = np.sum(x_weights * FP**2)

        if norm2_g_sq < 1e-15 or norm2_fg_sq < 1e-15:
            return 1e10, np.zeros(3)

        norm2_g = np.sqrt(norm2_g_sq)
        norm2_fg = np.sqrt(norm2_fg_sq)

        # L4 norms
        norm4_g_4 = np.sum(x_weights * P**4)
        norm4_fg_4 = np.sum(x_weights * FP**4)

        if norm4_g_4 < 1e-15 or norm4_fg_4 < 1e-15:
            return 1e10, np.zeros(3)

        norm4_g = norm4_g_4 ** 0.25
        norm4_fg = norm4_fg_4 ** 0.25

        # The bound on C4
        bound = (norm4_g * norm4_fg) / (norm2_g * norm2_fg)

        # Compute gradient using analytic derivatives
        da12_da = np.array([1.0, -3.0, 105.0]) / 10395.0

        H = np.array([H0, H4, H8])  # shape (3, n_quad)
        dP_da = H + da12_da[:, None] * H12[None, :]
        dFP_da = H - da12_da[:, None] * H12[None, :]

        # Derivatives of norms
        d_norm2_g_sq_da = 2.0 * (x_weights[None, :] * P[None, :] * dP_da).sum(axis=1)
        d_norm2_fg_sq_da = 2.0 * (x_weights[None, :] * FP[None, :] * dFP_da).sum(axis=1)
        d_norm4_g_4_da = 4.0 * (x_weights[None, :] * P[None, :]**3 * dP_da).sum(axis=1)
        d_norm4_fg_4_da = 4.0 * (x_weights[None, :] * FP[None, :]**3 * dFP_da).sum(axis=1)

        # d(norm2_g)/da_k = d_norm2_g_sq_da / (2 * norm2_g)
        d_norm2_g_da = d_norm2_g_sq_da / (2.0 * norm2_g)
        d_norm2_fg_da = d_norm2_fg_sq_da / (2.0 * norm2_fg)
        d_norm4_g_da = d_norm4_g_4_da / (4.0 * norm4_g)
        d_norm4_fg_da = d_norm4_fg_4_da / (4.0 * norm4_fg)

        # bound = (norm4_g * norm4_fg) / (norm2_g * norm2_fg)
        # log(bound) = log(norm4_g) + log(norm4_fg) - log(norm2_g) - log(norm2_fg)
        # d(bound)/da_k = bound * (d_norm4_g_da/norm4_g + d_norm4_fg_da/norm4_fg - d_norm2_g_da/norm2_g - d_norm2_fg_da/norm2_fg)

        grad = bound * (d_norm4_g_da / norm4_g + d_norm4_fg_da / norm4_fg
                        - d_norm2_g_da / norm2_g - d_norm2_fg_da / norm2_fg)

        return bound, grad

    def objective(coeffs):
        """Compute the upper bound on C4 for the given Hermite coefficients."""
        val, _ = compute_bound_and_gradient(coeffs)
        return val

    def grad_func(coeffs):
        _, grad = compute_bound_and_gradient(coeffs)
        return grad

    best_coeffs = seed.copy()
    best_val = objective(seed)

    # Phase 1: Differential evolution for global search
    # Define bounds around the seed
    bounds = [
        (seed[0] - 2.0, seed[0] + 2.0),
        (seed[1] - 0.2, seed[1] + 0.2),
        (seed[2] - 0.02, seed[2] + 0.02),
    ]

    if time.time() < deadline - 60.0:
        try:
            de_result = differential_evolution(
                objective,
                bounds=bounds,
                seed=42,
                maxiter=200,
                popsize=20,
                tol=1e-12,
                mut=(0.5, 1.5),
                recomb=0.8,
                polish=False,
                init='latinhypercube',
                update_interval=1,
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_coeffs = de_result.x.copy()
        except Exception:
            pass

    # Phase 2: L-BFGS-B local refinement with analytic gradient
    if time.time() < deadline - 20.0:
        try:
            lb_result = minimize(
                objective,
                best_coeffs,
                jac=grad_func,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'gtol': 1e-14, 'ftol': 1e-16}
            )
            if lb_result.fun < best_val:
                best_val = lb_result.fun
                best_coeffs = lb_result.x.copy()
        except Exception:
            pass

    # Phase 3: Final tight L-BFGS-B refinement
    if time.time() < deadline - 5.0:
        try:
            lb_result2 = minimize(
                objective,
                best_coeffs,
                jac=grad_func,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'gtol': 1e-16, 'ftol': 1e-18}
            )
            if lb_result2.fun < best_val:
                best_val = lb_result2.fun
                best_coeffs = lb_result2.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
