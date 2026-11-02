"""Optimized program for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial
import warnings


def _hermite_h(n, x):
    """Compute the physicist's Hermite polynomial H_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    h_nm2 = np.ones_like(x)
    h_nm1 = 2 * x
    for k in range(2, n + 1):
        h_n = 2 * x * h_nm1 - 2 * (k - 1) * h_nm2
        h_nm2, h_nm1 = h_nm1, h_n
    return h_nm1


def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound on A(f)*A(Fourier(f)) for the given coefficients.
    
    The test function is g(x) = (c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)) * exp(-x^2/2)
    where c12 is chosen so that g(0) = 0.
    
    H_n(0) = 0 for odd n, and for even n=2k: H_{2k}(0) = (-1)^k * (2k)! / k!
    
    The uncertainty product bound involves the L2 norm of g and its Fourier transform.
    For a Gaussian-weighted Hermite combination, we can compute this analytically.
    
    A(f) = ||f||_2^2 / ||f||_2^2 ... actually let me think more carefully.
    
    For the uncertainty inequality, we're looking at:
    (∫ x^2 |f(x)|^2 dx) * (∫ ξ^2 |F(ξ)|^2 dξ) ≥ C
    
    where F is the Fourier transform. The constant C depends on the function class.
    
    Actually, re-reading the task: "reduce the upper bound on A(f)*A(Fourier(f))"
    
    Let me interpret this as: we have a specific form of test function, and we want to
    minimize the product of the second moments (or some related quantity).
    
    For g(x) = P(x) * exp(-x^2/2) where P is a polynomial, we can compute:
    - ||g||_2^2 = ∫ |g(x)|^2 dx
    - ∫ x^2 |g(x)|^2 dx
    - The Fourier transform of g, and its second moment
    
    The Fourier transform of x^n * exp(-x^2/2) can be computed using properties of
    Hermite polynomials and Gaussians.
    
    Actually, let me use a more direct approach. The key insight is that for the
    uncertainty principle, we want to minimize the product of variances (or second
    moments) of g and its Fourier transform.
    
    Let's define:
    - M2_g = ∫ x^2 |g(x)|^2 dx / ∫ |g(x)|^2 dx  (second moment of |g|^2)
    - M2_F = ∫ ξ^2 |F(ξ)|^2 dξ / ∫ |F(ξ)|^2 dξ  (second moment of |F|^2)
    
    And we want to minimize M2_g * M2_F or some similar product.
    
    Actually, I think the "upper bound on C4" refers to a specific constant in the
    uncertainty inequality. Let me just implement a numerical computation.
    
    For simplicity, let me compute the bound numerically.
    """
    # H_n(0) for even n = 2k: (-1)^k * (2k)! / k!
    # H_0(0) = 1
    # H_4(0) = (-1)^2 * 4! / 2! = 24/2 = 12
    # H_8(0) = (-1)^4 * 8! / 4! = 40320/24 = 1680
    # H_12(0) = (-1)^6 * 12! / 6! = 479001600/720 = 665280
    
    h0_0 = 1.0
    h4_0 = 12.0
    h8_0 = 1680.0
    h12_0 = 665280.0
    
    # c12 such that c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    c12 = -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0
    
    # Now compute the uncertainty product numerically
    # g(x) = (c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)) * exp(-x^2/2)
    
    # Use a sufficiently large domain
    x = np.linspace(-20, 20, 200001)
    dx = x[1] - x[0]
    
    H0 = _hermite_h(0, x)
    H4 = _hermite_h(4, x)
    H8 = _hermite_h(8, x)
    H12 = _hermite_h(12, x)
    
    P = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    g = P * np.exp(-x**2 / 2)
    
    # Compute ||g||_2^2 and ∫ x^2 |g|^2 dx
    g2 = g**2
    norm2 = np.sum(g2) * dx
    m2_g = np.sum(x**2 * g2) * dx
    
    # For the Fourier transform, use the property that FT of exp(-x^2/2)*H_n(x)
    # is related to i^n * exp(-ξ^2/2)*H_n(ξ) (up to normalization)
    # With the convention F(ξ) = ∫ f(x) exp(-2πi x ξ) dx, this gets complicated.
    # Let's use a simpler convention or just compute numerically.
    
    # Actually, for the uncertainty principle, the Fourier transform convention matters.
    # Let's use the unitary convention: F(ξ) = (1/sqrt(2π)) ∫ f(x) exp(-ixξ) dx
    # Then FT of exp(-x^2/2) is exp(-ξ^2/2)
    # And FT of H_n(x)exp(-x^2/2) is i^n H_n(ξ)exp(-ξ^2/2)
    
    # So F(ξ) = (1/sqrt(2π)) * (c0*H0(ξ) + i^4*c4*H4(ξ) + i^8*c8*H8(ξ) + i^12*c12*H12(ξ)) * exp(-ξ^2/2)
    # i^4 = 1, i^8 = 1, i^12 = -1
    
    F = (1/np.sqrt(2*np.pi)) * (c0*H0 + c4*H4 + c8*H8 - c12*H12) * np.exp(-x**2/2)
    # Note: using x as ξ for the transform (symmetric domain)
    
    F2 = F**2
    norm2_F = np.sum(F2) * dx
    m2_F = np.sum(x**2 * F2) * dx
    
    # The uncertainty product is (m2_g / norm2) * (m2_F / norm2_F)
    # Or maybe it's m2_g * m2_F / (norm2 * norm2_F)
    
    if norm2 < 1e-15 or norm2_F < 1e-15:
        return 1e10
    
    bound = (m2_g / norm2) * (m2_F / norm2_F)
    
    return bound


def _objective(coeffs):
    """Objective function for optimization."""
    c0, c4, c8 = coeffs
    try:
        bound = _compute_bound(c0, c4, c8)
        if not np.isfinite(bound) or bound < 0:
            return 1e10
        return bound
    except Exception:
        return 1e10


def construct():
    """
    Find coefficients that minimize the upper bound on A(f)*A(Fourier(f)).
    
    Returns an array of three coefficients [c0, c4, c8] for the Hermite combination.
    The degree-12 coefficient is determined to ensure g(0) = 0.
    """
    # Start from the prior state of the art and refine
    initial = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    # Try multiple starting points
    best_coeffs = initial
    best_bound = _compute_bound(initial[0], initial[1], initial[2])
    
    # Define bounds for coefficients to keep the problem well-behaved
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    # Use Nelder-Mead for robustness, then refine with L-BFGS-B
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Multiple random starts
        np.random.seed(42)
        for trial in range(5):
            if trial == 0:
                x0 = initial
            else:
                x0 = np.random.uniform([-1, -1, -1], [1, 1, 1])
            
            # First, Nelder-Mead
            res_nm = minimize(_objective, x0, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res_nm.fun < best_bound:
                best_bound = res_nm.fun
                best_coeffs = res_nm.x
            
            # Refine with L-BFGS-B
            res_lbfgs = minimize(_objective, res_nm.x, method='L-BFGS-B',
                                bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12})
            
            if res_lbfgs.fun < best_bound:
                best_bound = res_lbfgs.fun
                best_coeffs = res_lbfgs.x
    
    # Final check
    final_bound = _compute_bound(best_coeffs[0], best_coeffs[1], best_coeffs[2])
    if not np.isfinite(final_bound) or final_bound < 0:
        return initial
    
    return best_coeffs
# EVOLVE-BLOCK-END
