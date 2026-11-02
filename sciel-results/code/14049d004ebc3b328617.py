"""Optimized program for uncertainty inequality coefficient search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import roots_hermite


def _hermite_poly(x, n):
    """Compute the nth physicists' Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    else:
        H0 = np.ones_like(x)
        H1 = 2 * x
        for k in range(2, n + 1):
            H2 = 2 * x * H1 - 2 * (k - 1) * H0
            H0, H1 = H1, H2
        return H1


def _compute_bound(coeffs):
    """
    Compute the upper bound on C4 given coefficients [a, b, c] for H0, H4, H8.
    The H12 coefficient is fixed by g(0) = 0.
    """
    a, b, c = coeffs
    
    # H_n(0) values for even n
    # H_0(0) = 1
    # H_4(0) = 3
    # H_8(0) = 105
    # H_12(0) = 10395
    d = -(a * 1.0 + b * 3.0 + c * 105.0) / 10395.0
    
    # We need to compute:
    # A(g) = sqrt(∫ x^2 |g(x)|^2 dx) / sqrt(∫ |g(x)|^2 dx) perhaps, or similar
    # For the uncertainty bound, it's likely related to the variance
    
    # Let's use Gauss-Hermite quadrature for integrals of P(x)^2 * exp(-x^2)
    # ∫ P(x)^2 exp(-x^2) dx and ∫ x^2 P(x)^2 exp(-x^2) dx
    
    # Gauss-Hermite quadrature: ∫ f(x) exp(-x^2) dx ≈ Σ w_i f(x_i)
    # We need enough points for a degree-24 polynomial (P is degree 12, P^2 is degree 24)
    n_quad = 50  # 50 points should be sufficient for degree 24
    
    x_quad, w_quad = roots_hermite(n_quad)
    
    # Evaluate P(x) = a*H0(x) + b*H4(x) + c*H8(x) + d*H12(x)
    P_x = (a * _hermite_poly(x_quad, 0) + 
           b * _hermite_poly(x_quad, 4) + 
           c * _hermite_poly(x_quad, 8) + 
           d * _hermite_poly(x_quad, 12))
    
    P2 = P_x ** 2
    
    # Integral of P^2 * exp(-x^2)
    norm2 = np.sum(w_quad * P2)
    
    # Integral of x^2 * P^2 * exp(-x^2)
    var_num = np.sum(w_quad * x_quad ** 2 * P2)
    
    if norm2 <= 0 or np.isclose(norm2, 0):
        return 1e10  # Penalize zero norm
    
    # The "spread" squared is var_num / norm2
    spread_sq = var_num / norm2
    
    # The bound on C4 is likely spread_sq * spread_sq (since position and momentum spreads are equal)
    # Or it could be spread_sq itself
    # Let's assume it's the product, which is spread_sq^2 since they're equal
    bound = spread_sq ** 2
    
    return bound


def construct():
    """Find optimal coefficients [a, b, c] minimizing the C4 upper bound."""
    start_time = time.time()
    deadline = start_time + 180.0  # Must finish within 180s
    
    # Seed solution
    best_coeffs = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_bound = _compute_bound(best_coeffs)
    
    # Try multiple starting points
    random_starts = [
        np.array([0, 0, 0]),
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
        np.array([1, 1, 1]),
        np.array([-1, -1, -1]),
        np.array([1, -1, 1]),
        np.array([-1, 1, -1]),
        np.array([0.5, 0.5, 0.5]),
        np.array([-0.5, -0.5, -0.5]),
    ]
    
    # Add the seed as a start too
    random_starts.append(best_coeffs.copy())
    
    for start in random_starts:
        if time.time() > deadline - 10:  # Leave 10s margin
            break
        
        try:
            result = minimize(
                _compute_bound,
                start,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            
            if result.success and result.fun < best_bound:
                best_bound = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    # Try L-BFGS-B with more precise convergence
    if time.time() < deadline - 30:
        for start in random_starts[:5]:
            if time.time() > deadline - 15:
                break
            try:
                result = minimize(
                    _compute_bound,
                    start,
                    method='L-BFGS-B',
                    options={'maxiter': 5000, 'ftol': 1e-15}
                )
                if result.fun < best_bound:
                    best_bound = result.fun
                    best_coeffs = result.x
            except Exception:
                pass
    
    # Final polish around the best solution
    if time.time() < deadline - 10:
        try:
            result = minimize(
                _compute_bound,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 50000, 'xatol': 1e-15, 'fatol': 1e-15}
            )
            if result.fun < best_bound:
                best_bound = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
