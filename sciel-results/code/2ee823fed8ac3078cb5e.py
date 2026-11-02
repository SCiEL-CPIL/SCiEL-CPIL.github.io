"""
Structural rewrite: Differential Evolution global optimization with local refinement
for the Hermite-Gaussian uncertainty bound minimization.

The key structural change: use scipy's differential_evolution (a robust global
optimizer) as the primary search, combined with Powell local refinement.
This avoids the multi-start Nelder-Mead approach which can get stuck in local
minima due to its simplex-based nature in 3D.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from numpy.polynomial.hermite import hermgauss


def construct():
    """Return three coefficients for H0, H4, H8 Hermite combination."""
    start_time = time.time()
    deadline = start_time + 195.0  # Leave 15s margin for safety

    # Seed from the problem description
    seed_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    # --- Mathematical Setup ---
    # g(x) = P(x) * exp(-x^2/2)
    # P(x) = a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)
    # Constraint: P(0) = 0 => a12 = (a0 - 3*a4 + 105*a8) / 10395
    # Fourier domain: Q(x) = a0*H0(x) + a4*H4(x) + a8*H8(x) - a12*H12(x)
    # Objective: (||P e^{-x^2/2}||_4 * ||Q e^{-x^2/2}||_4) / (||P e^{-x^2/2}||_2 * ||Q e^{-x^2/2}||_2)

    # --- Numerical Integration Setup ---
    N_QUAD = 40  # Use 40 nodes for high precision
    nodes, weights = hermgauss(N_QUAD)
    x_nodes = np.sqrt(2.0) * nodes
    w_scaled = np.sqrt(2.0) * weights

    def hermite_physicist(x, n):
        """Compute Physicists' Hermite polynomial H_n(x) using recurrence."""
        if n == 0:
            return np.ones_like(x, dtype=float)
        elif n == 1:
            return 2.0 * x
        else:
            H_nm1 = np.ones_like(x, dtype=float)
            H_n = 2.0 * x
            for k in range(2, n + 1):
                H_np1 = 2.0 * x * H_n - 2.0 * (k - 1) * H_nm1
                H_nm1 = H_n
                H_n = H_np1
            return H_n

    # Precompute Hermite polynomials at quadrature nodes
    H0_nodes = hermite_physicist(x_nodes, 0)
    H4_nodes = hermite_physicist(x_nodes, 4)
    H8_nodes = hermite_physicist(x_nodes, 8)
    H12_nodes = hermite_physicist(x_nodes, 12)

    def objective(coeffs):
        """Compute the uncertainty bound for given coefficients a0, a4, a8."""
        a0, a4, a8 = coeffs
        
        # Calculate a12 to satisfy P(0) = 0
        a12 = (a0 - 3.0 * a4 + 105.0 * a8) / 10395.0

        # Construct P(x) and Q(x) at nodes
        P_vals = a0 * H0_nodes + a4 * H4_nodes + a8 * H8_nodes + a12 * H12_nodes
        Q_vals = a0 * H0_nodes + a4 * H4_nodes + a8 * H8_nodes - a12 * H12_nodes

        # Compute squared L2 norms and fourth-power L4 norms
        norm2_g_sq = np.sum(w_scaled * P_vals**2)
        norm2_q_sq = np.sum(w_scaled * Q_vals**2)
        norm4_g_4 = np.sum(w_scaled * P_vals**4)
        norm4_q_4 = np.sum(w_scaled * Q_vals**4)

        # Avoid division by zero and invalid values
        if norm2_g_sq < 1e-12 or norm2_q_sq < 1e-12 or norm4_g_4 < 1e-12 or norm4_q_4 < 1e-12:
            return 1e10

        norm2_g = np.sqrt(norm2_g_sq)
        norm2_q = np.sqrt(norm2_q_sq)
        norm4_g = norm4_g_4 ** 0.25
        norm4_q = norm4_q_4 ** 0.25

        bound = (norm4_g * norm4_q) / (norm2_g * norm2_q)
        return bound

    # --- Global Optimization with Differential Evolution ---
    
    best_coeffs = seed_coeffs.copy()
    best_val = objective(seed_coeffs)

    # Define search bounds - expanded to explore more of the space
    bounds = [(-5.0, 5.0), (-3.0, 3.0), (-1.0, 1.0)]

    # Run differential evolution as the primary global search
    if time.time() < deadline - 30.0:
        try:
            de_result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=200,
                popsize=30,
                tol=1e-10,
                atol=1e-12,
                mutation=(0.5, 1.0),
                recombination=0.8,
                polish=False,  # We'll do our own local refinement
                update_interval=10
            )
            
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_coeffs = de_result.x.copy()
                
        except Exception:
            pass

    # Local refinement using Powell method (more robust than Nelder-Mead for this problem)
    if time.time() < deadline - 5.0:
        try:
            powell_result = minimize(
                objective,
                best_coeffs,
                method='Powell',
                options={'maxiter': 2000, 'xtol': 1e-14, 'ftol': 1e-16}
            )
            if powell_result.fun < best_val:
                best_val = powell_result.fun
                best_coeffs = powell_result.x.copy()
        except Exception:
            pass

    # Additional refinement with COBYLA (derivative-free, robust)
    if time.time() < deadline - 2.0:
        try:
            cobyla_result = minimize(
                objective,
                best_coeffs,
                method='COBYLA',
                options={'maxiter': 500, 'rhobeg': 1e-6, 'tol': 1e-14}
            )
            if cobyla_result.fun < best_val:
                best_val = cobyla_result.fun
                best_coeffs = cobyla_result.x.copy()
        except Exception:
            pass

    # Final check: also try from the seed point with local optimization
    if time.time() < deadline - 1.0:
        try:
            seed_result = minimize(
                objective,
                seed_coeffs,
                method='Powell',
                options={'maxiter': 1000, 'xtol': 1e-14, 'ftol': 1e-16}
            )
            if seed_result.fun < best_val:
                best_val = seed_result.fun
                best_coeffs = seed_result.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
