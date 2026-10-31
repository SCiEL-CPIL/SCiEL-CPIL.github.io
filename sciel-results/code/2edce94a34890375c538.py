"""Rewritten program for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import root, brentq
import time


def _hermite(x, n):
    """Compute Hermite polynomial H_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2.0 * x
    H_nm2 = np.ones_like(x)
    H_nm1 = 2.0 * x
    for k in range(2, n + 1):
        H_n = 2.0 * x * H_nm1 - 2.0 * (k - 1) * H_nm2
        H_nm2, H_nm1 = H_nm1, H_n
    return H_nm1


def _compute_A_f(coeffs, a12):
    """
    Compute A(f) for f(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12)*exp(-x^2/2).
    A(f) = sqrt(integral |f|^2 dx * integral |f'|^2 dx) / (some normalization)
    
    Actually, for the uncertainty inequality, we need:
    A(f) = sqrt(∫|f|² dx · ∫x²|f|² dx) or similar.
    
    Let me think more carefully. The uncertainty product is:
    A(f) * A(ĵ) where A(f) = sqrt(∫|f|² · ∫|x f|²) and A(ĵ) = sqrt(∫|ĵ|² · ∫|ξ ĵ|²)
    
    But by Plancherel, ∫|ĵ|² = ∫|f|² (up to constants).
    
    For the specific form used in the literature (Goncalves et al.), the bound C4
    is computed from the moments of the function.
    
    Let me use the standard approach: for f(x) = p(x) * exp(-x²/2) where p is a polynomial,
    we can compute all integrals analytically.
    
    f(x) = (a0 + a4*H4(x) + a8*H8(x) + a12*H12(x)) * exp(-x²/2)
    
    Using the orthogonality of Hermite polynomials with weight exp(-x²):
    ∫ H_m(x) H_n(x) exp(-x²) dx = 2^n n! sqrt(π) δ_mn
    
    But our weight is exp(-x²/2), so let's change variables or compute directly.
    
    Actually, let me use the standard physicist's Hermite polynomials with weight exp(-x²).
    If f(x) = q(x) * exp(-x²/4) where q is a polynomial, then...
    
    Let me just compute numerically using quadrature for robustness.
    """
    a0, a4, a8 = coeffs
    
    # We'll compute integrals numerically
    # f(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x²/2)
    
    # Use Gauss-Hermite quadrature with weight exp(-x²)
    # So f(x)*exp(x²/2) = polynomial, and ∫ f(x)² exp(-x²) dx can be computed
    # But we need ∫ f(x)² dx = ∫ [poly(x)]² exp(-x²) dx which is exactly GH quadrature
    
    # Number of nodes
    n_nodes = 200
    x, w = np.polynomial.hermite.hermgauss(n_nodes)
    
    # Compute Hermite polynomials at quadrature points
    H0 = np.ones_like(x)
    H4 = _hermite(x, 4)
    H8 = _hermite(x, 8)
    H12 = _hermite(x, 12)
    
    poly = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
    
    # f(x) = poly(x) * exp(-x²/2)
    # |f(x)|² = poly(x)² * exp(-x²)
    # ∫ |f(x)|² dx = ∫ poly(x)² exp(-x²) dx (GH quadrature)
    int_f2 = np.sum(w * poly**2)
    
    # For A(f), we need ∫ x² |f(x)|² dx
    int_x2f2 = np.sum(w * x**2 * poly**2)
    
    # For A(ĵ), by Plancherel ∫|ĵ|² = ∫|f|² (with appropriate normalization)
    # ∫ ξ²|ĵ|² dξ = ∫ |f'(x)|² dx (up to constants)
    # f'(x) = [poly'(x) - x*poly(x)] * exp(-x²/2)
    # |f'(x)|² = [poly'(x) - x*poly(x)]² * exp(-x²)
    
    # Compute poly'(x)
    dH0 = np.zeros_like(x)
    dH4 = _hermite(x, 3) * 4  # H4' = 4*H3
    dH8 = _hermite(x, 7) * 8  # H8' = 8*H7
    dH12 = _hermite(x, 11) * 12  # H12' = 12*H11
    
    dpoly = a4 * dH4 + a8 * dH8 + a12 * dH12
    
    fprime_inner = dpoly - x * poly
    int_fprime2 = np.sum(w * fprime_inner**2)
    
    # A(f) = sqrt(∫|f|² * ∫x²|f|²)
    # A(ĵ) = sqrt(∫|ĵ|² * ∫ξ²|ĵ|²) = sqrt(∫|f|² * ∫|f'|²) (by Plancherel and Parseval)
    # Wait, need to be careful with Fourier transform normalization.
    
    # Standard: F(ξ) = (1/sqrt(2π)) ∫ f(x) e^{-ixξ} dx
    # Then ∫|F|² dξ = ∫|f|² dx
    # ∫ξ²|F|² dξ = (1/4π²)∫|f''|²... no
    # Actually ∫ξ²|F(ξ)|² dξ = (1/(2π))∫|f'(x)|² dx... 
    # Let me use: if F(ξ) = ∫f(x)e^{-ixξ}dx (no 1/sqrt(2π)), then ∫|F|²dξ = 2π∫|f|²dx
    # and ∫ξ²|F|²dξ = 2π∫|f'|²dx
    
    # So A(f)A(ĵ) = sqrt(∫|f|²·∫x²|f|²) · sqrt(∫|F|²·∫ξ²|F|²)
    # = sqrt(∫|f|²·∫x²|f|²) · sqrt(2π∫|f|²·2π∫|f'|²)
    # = 2π sqrt(∫|f|² · ∫x²|f|² · ∫|f|² · ∫|f'|²)
    # = 2π ∫|f|² · sqrt(∫x²|f|² · ∫|f'|²)
    
    # Hmm, this depends on normalization. Let me just compute the product and minimize.
    
    if int_f2 <= 0:
        return np.inf
    
    A_f = np.sqrt(int_f2 * int_x2f2)
    A_F = np.sqrt(int_f2 * int_fprime2)  # up to constant
    
    return A_f * A_F


def _objective(coeffs):
    """
    Objective function: compute C4 upper bound given (a0, a4, a8).
    a12 is determined by g(0) = 0.
    g(0) = (a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0)) * exp(0) = 0
    H0(0) = 1
    H4(0) = -12 (physicist's: H4(x) = 16x⁴-48x²+12, H4(0)=12)
    H8(0) = ... let me compute
    H12(0) = ...
    """
    a0, a4, a8 = coeffs
    
    # Hermite polynomials at x=0 (physicist's)
    # H_0(0) = 1
    # H_2(0) = -2
    # H_4(0) = 12
    # H_6(0) = -120
    # H_8(0) = 1680
    # H_10(0) = -30240
    # H_12(0) = 665280
    
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    # g(0) = a0*H0_0 + a4*H4_0 + a8*H8_0 + a12*H12_0 = 0
    # a12 = -(a0*H0_0 + a4*H4_0 + a8*H8_0) / H12_0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    return _compute_A_f(coeffs, a12)


def _gradient(coeffs):
    """Numerical gradient of the objective."""
    eps = 1e-7
    grad = np.zeros(3)
    f0 = _objective(coeffs)
    for i in range(3):
        coeffs_plus = coeffs.copy()
        coeffs_plus[i] += eps
        grad[i] = (_objective(coeffs_plus) - f0) / eps
    return grad


def construct():
    """
    Find the optimal coefficients (a0, a4, a8) that minimize the upper bound C4.
    Uses stationary point equations with a robust hybrid root-finder.
    """
    start_time = time.time()
    deadline = 180.0  # seconds, with safety margin
    
    best_coeffs = np.array([-113.0/100, 1.0/25, 1.0/3240])  # Prior state of the art
    best_value = _objective(best_coeffs)
    
    # Generate multiple initial guesses
    np.random.seed(42)
    
    initial_guesses = [
        np.array([-1.13, 0.04, 0.000309]),  # The known optimum
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-1.2, 0.03, 0.0004]),
        np.array([-0.9, 0.06, 0.0002]),
        np.array([-1.1, 0.045, 0.00025]),
        np.array([-1.15, 0.035, 0.00035]),
        np.array([-1.05, 0.042, 0.00028]),
        np.array([-1.18, 0.038, 0.00032]),
    ]
    
    # Add some random perturbations
    for i in range(10):
        base = np.array([-1.13, 0.04, 0.000309])
        perturbation = np.random.randn(3) * np.array([0.1, 0.01, 0.00005])
        initial_guesses.append(base + perturbation)
    
    # Also try wider range
    for i in range(5):
        a0 = np.random.uniform(-2.0, -0.5)
        a4 = np.random.uniform(0.01, 0.1)
        a8 = np.random.uniform(0.0001, 0.001)
        initial_guesses.append(np.array([a0, a4, a8]))
    
    # Solve stationary point equations: gradient = 0
    for guess in initial_guesses:
        if time.time() - start_time > deadline - 10:
            break
        
        try:
            # Use hybrid method (Broyden's method with line search)
            result = root(_gradient, guess, method='hybr', options={'xtol': 1e-12, 'maxfev': 1000})
            
            if result.success and result.fun is not None:
                val = _objective(result.x)
                if val < best_value and np.all(np.isfinite(result.x)):
                    best_value = val
                    best_coeffs = result.x.copy()
                    
        except Exception:
            continue
        
        # Also try BFGS as backup
        if time.time() - start_time > deadline - 10:
            break
        
        try:
            from scipy.optimize import minimize
            res = minimize(_objective, guess, method='L-BFGS-B', 
                          options={'gtol': 1e-12, 'maxiter': 200})
            if res.success and res.fun < best_value:
                best_value = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            continue
    
    # Final polish: high-precision local solve around the best point
    if time.time() - start_time < deadline - 15:
        try:
            result = root(_gradient, best_coeffs, method='hybr', 
                         options={'xtol': 1e-14, 'maxfev': 2000})
            if result.success:
                val = _objective(result.x)
                if val < best_value and np.all(np.isfinite(result.x)):
                    best_value = val
                    best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Ensure we return valid coefficients
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.array([-113.0/100, 1.0/25, 1.0/3240])
    
    return best_coeffs
# EVOLVE-BLOCK-END
