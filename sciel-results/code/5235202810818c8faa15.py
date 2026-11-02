"""
Optimized program: numerical optimization of Hermite coefficients for the uncertainty bound.
Combines Gauss-Hermite quadrature (exact for polynomial-Gaussian integrals) with
differential evolution for global search and local refinement.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from numpy.polynomial.hermite import hermgauss


def construct():
    """Return three coefficients for H0, H4, H8 Hermite combination."""
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin for 210s wall-clock limit

    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    # Gauss-Hermite quadrature nodes and weights
    # We need N=25 for exactness of degree-48 polynomials (P^4 where P is deg 12)
    nodes, weights = hermgauss(25)
    sqrt2 = np.sqrt(2.0)
    x_nodes = sqrt2 * nodes

    def get_a12(a0, a4, a8):
        """Compute a12 so that g(0) = 0."""
        # H0(0)=1, H4(0)=-3, H8(0)=105, H12(0)=-10395
        # g(0) = (a0*1 + a4*(-3) + a8*105 + a12*(-10395)) = 0
        return (a0 - 3.0 * a4 + 105.0 * a8) / 10395.0

    def hermite_p(x, n):
        """Compute physicists' Hermite polynomial H_n(x) for n in {0,4,8,12}."""
        if n == 0:
            return np.ones_like(x)
        elif n == 4:
            return 16.0 * x**4 - 48.0 * x**2 + 12.0
        elif n == 8:
            return 256.0 * x**8 - 2304.0 * x**6 + 5760.0 * x**4 - 4320.0 * x**2 + 504.0
        elif n == 12:
            return (4096.0 * x**12 - 55296.0 * x**10 + 303104.0 * x**8
                    - 589824.0 * x**6 + 408192.0 * x**4 - 77760.0 * x**2 + 10395.0)
        else:
            raise ValueError(f"Unsupported n={n}")

    # Precompute Hermite polynomials at quadrature nodes
    H0_nodes = np.ones_like(x_nodes)
    H4_nodes = hermite_p(x_nodes, 4)
    H8_nodes = hermite_p(x_nodes, 8)
    H12_nodes = hermite_p(x_nodes, 12)

    def compute_poly_values(coeffs):
        """
        Compute P(x_nodes) and Q(x_nodes) for given coeffs.
        P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
        Q(x) = a0*H0 + a4*H4 + a8*H8 - a12*H12  (Fourier transform poly part)
        """
        a0, a4, a8 = coeffs
        a12 = get_a12(a0, a4, a8)
        
        P_vals = a0 * H0_nodes + a4 * H4_nodes + a8 * H8_nodes + a12 * H12_nodes
        Q_vals = a0 * H0_nodes + a4 * H4_nodes + a8 * H8_nodes - a12 * H12_nodes
        
        return P_vals, Q_vals

    def objective(coeffs):
        """
        Compute the upper bound on C4.
        """
        P_vals, Q_vals = compute_poly_values(coeffs)
        
        # Norms using Gauss-Hermite quadrature
        # ||g||_2^2 = sqrt(2) * sum(w * P^2)
        norm2_g_sq = sqrt2 * np.sum(weights * P_vals**2)
        norm2_q_sq = sqrt2 * np.sum(weights * Q_vals**2)
        
        # ||g||_4^4 = sqrt(2) * sum(w * P^4)
        norm4_g_4 = sqrt2 * np.sum(weights * P_vals**4)
        norm4_q_4 = sqrt2 * np.sum(weights * Q_vals**4)
        
        if norm2_g_sq < 1e-15 or norm2_q_sq < 1e-15 or norm4_g_4 < 1e-15 or norm4_q_4 < 1e-15:
            return 1e10
            
        norm2_g = np.sqrt(norm2_g_sq)
        norm2_q = np.sqrt(norm2_q_sq)
        norm4_g = norm4_g_4 ** 0.25
        norm4_q = norm4_q_4 ** 0.25
        
        bound = (norm4_g * norm4_q) / (norm2_g * norm2_q)
        return bound

    # Initial evaluation
    best_coeffs = seed.copy()
    best_val = objective(seed)
    
    # Differential Evolution for global search
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.01, 0.01)]
    
    if time.time() < deadline - 60.0:
        try:
            de_result = differential_evolution(
                objective,
                bounds=bounds,
                seed=42,
                maxiter=300,
                tol=1e-10,
                polish=False,
                init='sobol'
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_coeffs = de_result.x.copy()
        except Exception:
            pass

    # Local refinement with Nelder-Mead
    if time.time() < deadline - 15.0:
        try:
            result = minimize(
                objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
            
    # One more polish with Powell if time remains
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                objective,
                best_coeffs,
                method='Powell',
                options={'maxiter': 10000, 'xtol': 1e-14, 'ftol': 1e-16}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
