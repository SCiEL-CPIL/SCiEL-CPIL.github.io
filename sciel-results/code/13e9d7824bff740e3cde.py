"""
Rewritten program: numerical optimization of Hermite coefficients for the uncertainty bound.
Strategy:
1. Derive the correct objective: product of L4 norms divided by product of L2 norms.
2. Use Gauss-Hermite quadrature for exact integration of the polynomial-Gaussian integrals.
3. Perform a global search using Differential Evolution (DE).
4. Refine using a local optimizer (Nelder-Mead or L-BFGS-B).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from numpy.polynomial.hermite import hermgauss


def construct():
    """Return three coefficients for H0, H4, H8 Hermite combination."""
    start_time = time.time()
    # We have 210s wall clock. Leave margin for overhead and return.
    deadline = start_time + 190.0

    # Seed from the problem description or a known good starting point
    # The prompt mentions "seed program" score 1.0.
    # The current program's seed was: [-113/100, 1/25, 1/3240]
    seed_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    # --- Mathematical Setup ---
    # We are looking for coefficients a0, a4, a8 for the function:
    # g(x) = P(x) * exp(-x^2/2)
    # where P(x) = a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)
    # Constraint: P(0) = 0 => a0 - 3*a4 + 105*a8 - 10395*a12 = 0
    # (Note: Physicists' Hermite polynomials H_n(0) are: H0(0)=1, H4(0)=-3, H8(0)=105, H12(0)=-10395)
    # So a12 = (a0 - 3*a4 + 105*a8) / 10395

    # The Fourier transform of H_n(x) exp(-x^2/2) involves (-i)^n H_n(x) exp(-x^2/2) (up to normalization).
    # Wait, the standard unitary Fourier transform F(f)(k) = (1/sqrt(2pi)) int f(x) e^{-ikx} dx.
    # If f(x) = H_n(x) e^{-x^2/2}, then F(f)(k) = (-i)^n H_n(k) e^{-k^2/2}.
    # So the polynomial part Q(k) in the Fourier domain is:
    # Q(k) = a0*H0(k) + a4*(-i)^4 H4(k) + a8*(-i)^8 H8(k) + a12*(-i)^12 H12(k)
    # (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = -1.
    # So Q(k) = a0*H0(k) + a4*H4(k) + a8*H8(k) - a12*H12(k).

    # The quantity to minimize is the upper bound on A(f)*A(F(f)).
    # Based on the prompt "uncertainty inequality" and "Fourier-invariant",
    # and the current program's logic, the objective is likely:
    # Bound = (||g||_4 * ||F(g)||_4) / (||g||_2 * ||F(g)||_2)
    # Note: ||F(g)||_p = ||g||_p due to Plancherel and scaling invariance of Lp norms for unitary FT?
    # Actually, Plancherel says ||F(g)||_2 = ||g||_2.
    # For L4, it is not invariant. But the problem asks to reduce the upper bound.
    # Let's look at the current program's objective:
    # bound = (norm4_g * norm4_q) / (norm2_g * norm2_q)
    # where norm4_g is ||P e^{-x^2/2}||_4 and norm4_q is ||Q e^{-x^2/2}||_4.
    # This matches the structure of an uncertainty product where one factor comes from x-space and one from k-space.

    # --- Numerical Integration Setup ---
    # We need to compute:
    # I2 = int P(x)^2 e^{-x^2/2} dx
    # I4 = int P(x)^4 e^{-x^2/2} dx
    # J2 = int Q(x)^2 e^{-x^2/2} dx
    # J4 = int Q(x)^4 e^{-x^2/2} dx
    
    # Using Gauss-Hermite quadrature:
    # int f(x) e^{-x^2} dx ~ sum w_i f(x_i)
    # Our weight is e^{-x^2/2}. Let x = sqrt(2) t. dx = sqrt(2) dt.
    # e^{-x^2/2} = e^{-t^2}.
    # Integral = sqrt(2) int f(sqrt(2) t) e^{-t^2} dt.
    # So we can use standard hermgauss nodes/weights (which integrate against e^{-t^2}).
    # We need to evaluate f at x_i = sqrt(2) * t_i.
    
    # Degree of P and Q is 12.
    # P^4 is degree 48.
    # Q^4 is degree 48.
    # We need 2N - 1 >= 48 => N >= 24.5 => N = 25.
    # Let's use N=30 for safety and precision.
    
    N_QUAD = 30
    nodes, weights = hermgauss(N_QUAD)
    x_nodes = np.sqrt(2.0) * nodes
    # The integral is sqrt(2) * sum(weights * f(x_nodes))
    sqrt2 = np.sqrt(2.0)
    w_scaled = sqrt2 * weights

    def hermite_physicist(x, n):
        """Compute Physicists' Hermite polynomial H_n(x)."""
        # Recurrence: H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
        if n == 0:
            return np.ones_like(x)
        elif n == 1:
            return 2.0 * x
        else:
            H_nm1 = np.ones_like(x)
            H_n = 2.0 * x
            for k in range(2, n + 1):
                H_np1 = 2.0 * x * H_n - 2.0 * (k - 1) * H_nm1
                H_nm1 = H_n
                H_n = H_np1
            return H_n

    # Precompute Hermite polynomials at nodes
    H0_nodes = hermite_physicist(x_nodes, 0)
    H4_nodes = hermite_physicist(x_nodes, 4)
    H8_nodes = hermite_physicist(x_nodes, 8)
    H12_nodes = hermite_physicist(x_nodes, 12)

    def objective(coeffs):
        """
        Compute the uncertainty bound for given coefficients a0, a4, a8.
        """
        a0, a4, a8 = coeffs
        
        # Calculate a12 to satisfy P(0) = 0
        # H0(0)=1, H4(0)=-3, H8(0)=105, H12(0)=-10395
        a12 = (a0 - 3.0 * a4 + 105.0 * a8) / 10395.0

        # Construct P(x) and Q(x) at nodes
        # P(x) = a0 H0 + a4 H4 + a8 H8 + a12 H12
        # Q(x) = a0 H0 + a4 H4 + a8 H8 - a12 H12  (because (-i)^12 = -1)
        
        P_vals = a0 * H0_nodes + a4 * H4_nodes + a8 * H8_nodes + a12 * H12_nodes
        Q_vals = a0 * H0_nodes + a4 * H4_nodes + a8 * H8_nodes - a12 * H12_nodes

        # Compute norms
        # ||g||_2^2 = sqrt(2) sum w_i P(x_i)^2
        norm2_g_sq = np.sum(w_scaled * P_vals**2)
        norm2_q_sq = np.sum(w_scaled * Q_vals**2)
        
        # ||g||_4^4 = sqrt(2) sum w_i P(x_i)^4
        norm4_g_4 = np.sum(w_scaled * P_vals**4)
        norm4_q_4 = np.sum(w_scaled * Q_vals**4)

        # Avoid division by zero
        if norm2_g_sq < 1e-12 or norm2_q_sq < 1e-12 or norm4_g_4 < 1e-12 or norm4_q_4 < 1e-12:
            return 1e10

        norm2_g = np.sqrt(norm2_g_sq)
        norm2_q = np.sqrt(norm2_q_sq)
        norm4_g = np.sqrt(np.sqrt(norm4_g_4))  # fourth root
        norm4_q = np.sqrt(np.sqrt(norm4_q_4))

        # The bound
        bound = (norm4_g * norm4_q) / (norm2_g * norm2_q)
        return bound

    # --- Optimization ---
    
    best_coeffs = seed_coeffs.copy()
    best_val = objective(seed_coeffs)

    # Define bounds for the search space
    # The coefficients should be of moderate magnitude.
    # Let's explore a range around the seed.
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-1.0, 1.0)]

    # 1. Differential Evolution for global search
    if time.time() < deadline - 30.0:
        try:
            de_result = differential_evolution(
                objective,
                bounds=bounds,
                seed=42,
                maxiter=300,
                tol=1e-9,
                polish=False, # We will polish manually
                init='sobol',
                popsize=15
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_coeffs = de_result.x.copy()
        except Exception as e:
            print(f"DE failed: {e}")

    # 2. Local Refinement
    # Try Nelder-Mead first as it doesn't require gradients
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception as e:
            print(f"NM failed: {e}")

    # Try L-BFGS-B if time remains (might be faster if gradient was implemented, but we use numerical)
    # Actually, L-BFGS-B with numerical gradient might be less robust than NM for this non-linear, non-convex function.
    # Let's try Powell which is also derivative-free.
    if time.time() < deadline - 2.0:
        try:
            result = minimize(
                objective,
                best_coeffs,
                method='Powell',
                options={'maxiter': 2000, 'xtol': 1e-12, 'ftol': 1e-14}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x.copy()
        except Exception as e:
            print(f"Powell failed: {e}")

    # Final check: return the best coefficients found
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
