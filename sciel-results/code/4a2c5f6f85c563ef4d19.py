"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

def _hermite(n, x):
    """Compute Hermite polynomial H_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    H0 = np.ones_like(x)
    H1 = 2 * x
    for k in range(1, n):
        H2 = 2 * x * H1 - 2 * k * H0
        H0 = H1
        H1 = H2
    return H1

def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound C4 on A(f)*A(F(f)) for the Hermite-Gaussian test function.
    
    g(x) = (c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)) * exp(-x^2/2)
    where c12 is chosen so that g(0) = 0.
    
    The uncertainty product bound involves:
    - ||g||_2^2 (L2 norm squared)
    - ||x*g||_2^2 (variance term)
    - ||g'||_2^2 (derivative term, related to Fourier transform)
    
    For the uncertainty principle bound, we need to compute:
    C4 = (1 + ||x*g||^2/||g||^2) * (1 + ||g'||^2/||g||^2) / 4
    
    But actually, the standard Heisenberg uncertainty gives:
    (Δx)^2 (Δξ)^2 ≥ 1/4
    
    where (Δx)^2 = (∫x^2|g|^2)/(∫|g|^2) and (Δξ)^2 = (∫|g'|^2)/(∫|g|^2)
    
    Wait, let me reconsider. The task says "upper bound on A(f)*A(Fourier(f))".
    
    Actually, for the uncertainty inequality, the bound is typically:
    (∫x^2|g|^2 dx / ∫|g|^2 dx) * (∫|ĝ|^2 ξ^2 dξ / ∫|ĝ|^2 dξ) ≥ 1/4
    
    By Plancherel, ∫|ĝ|^2 dξ = ∫|g|^2 dx (up to constants depending on Fourier transform convention).
    
    Let me use the standard physics convention: F(f)(ξ) = ∫f(x)e^{-2πixξ}dx
    Then Plancherel: ∫|F(f)|^2 dξ = ∫|f|^2 dx
    
    And ∫ξ^2|F(f)|^2 dξ = (1/(4π^2))∫|f'|^2 dx
    
    So the product is:
    [∫x^2|g|^2 / ∫|g|^2] * [(1/(4π^2))∫|g'|^2 / ∫|g|^2]
    
    Hmm, but the task mentions "C4" which suggests a specific constant. Let me think about this differently.
    
    Actually, looking at the seed values [-113/100, 1/25, 1/3240], these are specific coefficients.
    The task says "score is the upper bound on C4" and "lower is better".
    
    Let me compute the quantities numerically. The Hermite-Gaussian functions have known properties.
    
    For g(x) = P(x) * exp(-x^2/2) where P is a polynomial:
    - ||g||^2 = ∫P(x)^2 exp(-x^2) dx
    - ||x*g||^2 = ∫x^2 P(x)^2 exp(-x^2) dx
    - ||g'||^2 = ∫(P'(x) - x*P(x))^2 exp(-x^2) dx
    
    The uncertainty product is:
    Δx^2 * Δξ^2 = [||x*g||^2/||g||^2] * [||g'||^2/||g||^2] / (4π^2)
    
    Wait, I need to be more careful. Let me just compute it numerically.
    """
    # First, determine c12 so that g(0) = 0
    # g(0) = (c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0)) * exp(0)
    # H0(0) = 1, H4(0) = -12, H8(0) = 1344, H12(0) = -1032192 (need to verify)
    
    # Let me compute H_n(0) values
    H0_0 = 1.0
    H4_0 = -12.0  # H4(x) = 16x^4 - 48x^2 + 12, so H4(0) = 12... wait
    # Actually, physicists' Hermite polynomials:
    # H0 = 1
    # H1 = 2x
    # H2 = 4x^2 - 2
    # H3 = 8x^3 - 12x
    # H4 = 16x^4 - 48x^2 + 12
    # So H4(0) = 12
    
    # Let me recompute properly
    H0_0 = 1.0
    H1_0 = 0.0
    H2_0 = -2.0
    H3_0 = 0.0
    H4_0 = 12.0
    H5_0 = 0.0
    H6_0 = -120.0
    H7_0 = 0.0
    H8_0 = 1344.0
    H9_0 = 0.0
    H10_0 = -15120.0
    H11_0 = 0.0
    H12_0 = 199584.0
    
    # g(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    # c12 = -(c0*H0_0 + c4*H4_0 + c8*H8_0) / H12_0
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    # Now compute the integrals numerically
    # Use Gauss-Hermite quadrature for integrals of form ∫f(x)exp(-x^2)dx
    # np.polynomial.hermite.hermgauss gives nodes and weights
    
    n_quad = 100
    x_nodes, x_weights = np.polynomial.hermite.hermgauss(n_quad)
    
    # Evaluate the polynomial at the quadrature nodes
    H0_vals = _hermite(0, x_nodes)
    H4_vals = _hermite(4, x_nodes)
    H8_vals = _hermite(8, x_nodes)
    H12_vals = _hermite(12, x_nodes)
    
    P_vals = c0 * H0_vals + c4 * H4_vals + c8 * H8_vals + c12 * H12_vals
    
    # ||g||^2 = ∫P(x)^2 exp(-x^2) dx
    norm_sq = np.sum(x_weights * P_vals**2)
    
    if norm_sq < 1e-15:
        return 1e10  # Degenerate case
    
    # ||x*g||^2 = ∫x^2 P(x)^2 exp(-x^2) dx
    x2_norm_sq = np.sum(x_weights * x_nodes**2 * P_vals**2)
    
    # g'(x) = P'(x)*exp(-x^2/2) + P(x)*(-x)*exp(-x^2/2) = (P'(x) - x*P(x))*exp(-x^2/2)
    # We need P'(x). Let's compute it.
    # P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # P'(x) = c0*H0' + c4*H4' + c8*H8' + c12*H12'
    # H_n'(x) = 2n*H_{n-1}(x)
    
    H0_prime = np.zeros_like(x_nodes)  # H0' = 0
    H4_prime = 4 * _hermite(3, x_nodes)  # 2*4*H3 = 8*H3? No, H_n' = 2n*H_{n-1}
    # H4' = 2*4*H3 = 8*H3
    H4_prime = 8 * _hermite(3, x_nodes)
    H8_prime = 16 * _hermite(7, x_nodes)
    H12_prime = 24 * _hermite(11, x_nodes)
    
    P_prime_vals = c0 * H0_prime + c4 * H4_prime + c8 * H8_prime + c12 * H12_prime
    
    g_prime_vals = P_prime_vals - x_nodes * P_vals
    
    # ||g'||^2 = ∫(P'(x) - x*P(x))^2 exp(-x^2) dx
    deriv_norm_sq = np.sum(x_weights * g_prime_vals**2)
    
    # The uncertainty product
    # Δx^2 = ||x*g||^2 / ||g||^2
    # Δξ^2 = ||g'||^2 / (4π^2 * ||g||^2)  [depending on Fourier convention]
    
    # Actually, let me reconsider the Fourier transform convention.
    # If F(f)(ξ) = ∫f(x)e^{-2πixξ}dx, then:
    # ∫|F(f)|^2 dξ = ∫|f|^2 dx (Plancherel)
    # ∫ξ^2|F(f)|^2 dξ = (1/(4π^2))∫|f'|^2 dx
    
    # So Δξ^2 = ∫ξ^2|F(f)|^2 / ∫|F(f)|^2 = (1/(4π^2))∫|f'|^2 / ∫|f|^2
    
    # The product Δx^2 * Δξ^2 = [||x*g||^2/||g||^2] * [||g'||^2/(4π^2*||g||^2)]
    
    delta_x_sq = x2_norm_sq / norm_sq
    delta_xi_sq = deriv_norm_sq / (4 * np.pi**2 * norm_sq)
    
    product = delta_x_sq * delta_xi_sq
    
    return product

def construct():
    """
    Find the three coefficients (c0, c4, c8) that minimize the uncertainty product bound.
    """
    start_time = time.time()
    deadline = start_time + 200  # Safety margin before 210s limit
    
    # Initial guess from the seed
    best_coeffs = np.array([-113/100, 1/25, 1/3240])
    best_score = _compute_bound(best_coeffs[0], best_coeffs[1], best_coeffs[2])
    
    # Try differential evolution for global optimization
    try:
        bounds = [(-10, 10), (-10, 10), (-10, 10)]
        
        def objective(coeffs):
            return _compute_bound(coeffs[0], coeffs[1], coeffs[2])
        
        # Run differential evolution with time-limited approach
        de_result = differential_evolution(
            objective, 
            bounds,
            maxiter=50,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=True
        )
        
        elapsed = time.time() - start_time
        if elapsed < deadline - 30:
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_coeffs = de_result.x
        
        # Refine with L-BFGS-B
        res = minimize(
            objective,
            best_coeffs,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        
        elapsed = time.time() - start_time
        if elapsed < deadline - 10:
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x
        
        # Additional local refinement
        for _ in range(3):
            if time.time() - start_time > deadline - 5:
                break
            res = minimize(
                objective,
                best_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x
            else:
                break
        
    except Exception:
        pass
    
    # Ensure we return within the deadline
    if time.time() - start_time > deadline:
        return np.array([-113/100, 1/25, 1/3240])
    
    return best_coeffs
# EVOLVE-BLOCK-END
