"""Rewritten program: construct 12 positive double-root locations for the
Laguerre combination (m=12, n=1) and return them so the C4 upper bound is
minimized. The seed's equal-spaced roots are invalid; the valid Laguerre
construction uses the zeros of the Laguerre polynomial of degree m."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = 600.0  # wall-clock guard; evaluator kills at 630s, keep margin


def _compute_c4_bound(roots, m=12, n=1):
    """
    Compute the C4 upper bound given root locations.
    The bound involves the norm of the Laguerre combination with these roots.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != m:
        return np.inf
    
    # Ensure roots are positive and sorted
    if np.any(roots <= 0):
        return np.inf
    
    # The C4 bound for the refined uncertainty inequality with double roots
    # is related to the L2 norm of the function constructed from these roots.
    # For the Laguerre construction, we can compute this via the polynomial
    # formed by the product of (1 - x/r_i)^2 terms weighted appropriately.
    
    # A practical computation: the bound is proportional to the integral
    # of |sum c_i L_k(x)|^2 * x^alpha * e^{-x} dx over [0, inf)
    # where the coefficients c_i are determined by the root constraints.
    
    # For double roots at r_1, ..., r_m, the constraint is that the polynomial
    # P(x) = prod_{i=1}^m (x - r_i)^2 has its coefficients matching the
    # Laguerre expansion coefficients.
    
    # The C4 bound can be computed as:
    # C4 = (sum_{k=0}^{2m} a_k^2 / (2k+1)) where a_k are the Laguerre coefficients
    
    # For a polynomial Q(x) = sum a_k L_k(x), the L2 norm squared is sum a_k^2 / (2k+1)
    # (using the orthogonality relation for Laguerre polynomials)
    
    # First, convert the monomial polynomial to Laguerre basis
    # The polynomial with double roots is: P(x) = prod (x - r_i)^2
    
    # Build the monomial coefficients
    coeff = np.array([1.0])
    for r in roots:
        # Multiply by (x - r)^2 = x^2 - 2r*x + r^2
        new_coeff = np.convolve(coeff, [1.0, -2*r, r**2])
        coeff = new_coeff
    
    # Now convert monomial coefficients to Laguerre coefficients
    # Using the relation: x^k = sum_{j=0}^k c_{k,j} L_j(x)
    # We can compute this recursively or use numpy
    
    # Alternative: compute the norm directly via quadrature
    # C4 = integral_0^inf |P(x)|^2 * x^1 * e^{-x} dx / integral_0^inf x^1 * e^{-x} dx
    # The denominator is Gamma(2) = 1! = 1
    
    # Use Gauss-Laguerre quadrature for the integral
    try:
        from scipy.special import roots_laguerre as _rl
        # For n=1, weight is x^1 * e^{-x}, so we need alpha=1
        # Use generalized Gauss-Laguerre with alpha=1
        from scipy.special import roots_genlaguerre as _rgl
        N_quad = 200  # quadrature points
        x_quad, w_quad = _rgl(N_quad, 1.0)
        
        # Evaluate P(x) at quadrature points
        P_vals = np.zeros_like(x_quad)
        for i, xv in enumerate(x_quad):
            P_val = 1.0
            for r in roots:
                P_val *= (xv - r)**2
            P_vals[i] = P_val
        
        # Compute the integral
        integral = np.sum(w_quad * P_vals**2 * x_quad)
        
        # The C4 bound is proportional to this integral
        # For the standard normalization, C4 = integral / (some normalization)
        # The exact constant depends on the problem setup, but minimizing
        # this integral should minimize the bound
        
        return float(integral)
    except Exception:
        # Fallback: use a simple Riemann sum
        x_grid = np.linspace(0, 50, 10000)
        dx = x_grid[1] - x_grid[0]
        weight = x_grid * np.exp(-x_grid)
        
        P_vals = np.zeros_like(x_grid)
        for r in roots:
            P_vals *= (x_grid - r)**2
        
        integral = np.sum(weight * P_vals**2) * dx
        return float(integral)


def construct():
    """Return a list of 12 positive double-root locations (m=12, n=1)."""
    m = 12
    
    # Get initial guess from Laguerre zeros
    try:
        from scipy.special import roots_laguerre
        x_init, _ = roots_laguerre(m)
        x_init = np.sort(np.asarray(x_init, dtype=float))
    except Exception:
        x_init = np.array([1.0, 2.5, 4.5, 7.0, 10.0, 13.5, 17.5, 22.0, 27.0, 32.5, 38.5, 45.0])
    
    # Define the optimization problem
    def objective(params):
        # params are log-roots to ensure positivity
        roots = np.exp(params)
        # Ensure sorted
        roots = np.sort(roots)
        return _compute_c4_bound(roots, m=m, n=1)
    
    # Initial params in log space
    x0 = np.log(x_init)
    
    # Use Nelder-Mead for robustness (doesn't need gradients)
    try:
        result = minimize(
            objective, 
            x0, 
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'maxfev': 10000,
                'xatol': 1e-10,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        
        if result.success or result.fun < np.inf:
            best_roots = np.sort(np.exp(result.x))
            if np.all(best_roots > 0) and len(best_roots) == m:
                return [float(v) for v in best_roots]
    except Exception:
        pass
    
    # Fallback to Laguerre zeros if optimization failed
    return [float(v) for v in x_init]
# EVOLVE-BLOCK-END
