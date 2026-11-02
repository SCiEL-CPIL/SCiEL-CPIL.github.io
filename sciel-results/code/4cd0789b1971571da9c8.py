"""Rewritten program: Optimizes Hermite coefficients using Differential Evolution
with high-resolution grid for sup-norm and adaptive quadrature for L2-norm normalization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_hermite
from scipy.optimize import differential_evolution
from scipy.integrate import quad

# Pre-compute Hermite polynomials on a dense grid for speed
# The grid needs to cover the support of the Gaussian.
# e^{-x^2/2} is negligible beyond |x| > 10.
X_GRID = np.linspace(-12.0, 12.0, 50000)
H0 = np.ones_like(X_GRID)
H4 = eval_hermite(4, X_GRID)
H8 = eval_hermite(8, X_GRID)
H12 = eval_hermite(12, X_GRID)
GAUSS = np.exp(-0.5 * X_GRID * X_GRID)

# H values at 0 for constraint
H4_0 = -12.0
H8_0 = 168.0
H12_0 = 792.0

def _compute_g_and_gt(c0, c1, c2):
    """Compute g(x) and gt(x) on the pre-computed grid."""
    c3 = -(c0 + c1 * H4_0 + c2 * H8_0) / H12_0
    poly = c0 * H0 + c1 * H4 + c2 * H8 + c3 * H12
    g = poly * GAUSS
    # For this specific combination, Fourier transform is identical to g
    # because phases (-i)^4k = 1.
    gt = g
    return g, gt

def _l2_norm(c0, c1, c2):
    """Compute L2 norm using adaptive quadrature."""
    def integrand(x):
        c3 = -(c0 + c1 * H4_0 + c2 * H8_0) / H12_0
        # Evaluate Hermite polynomials at x
        h0 = 1.0
        h4 = eval_hermite(4, x)
        h8 = eval_hermite(8, x)
        h12 = eval_hermite(12, x)
        poly = c0 * h0 + c1 * h4 + c2 * h8 + c3 * h12
        val = poly * np.exp(-0.5 * x * x)
        return val * val
    
    # Integrate from -inf to inf. Use limits where Gaussian is small.
    try:
        norm_sq, _ = quad(integrand, -15.0, 15.0, epsabs=1e-12, epsrel=1e-12)
        if norm_sq <= 0:
            return 0.0
        return np.sqrt(norm_sq)
    except Exception:
        return 0.0

def _objective(params):
    c0, c1, c2 = params
    g, gt = _compute_g_and_gt(c0, c1, c2)
    
    # Check if function is effectively zero
    max_g = np.max(np.abs(g))
    if max_g < 1e-10:
        return 1e10 # Penalize zero function
    
    norm = _l2_norm(c0, c1, c2)
    if norm < 1e-10:
        return 1e10
        
    # Normalized sup norms
    a_g = max_g / norm
    a_gt = np.max(np.abs(gt)) / norm # norm is same for gt due to Plancherel
    
    return a_g * a_gt

def construct():
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    # Differential Evolution is robust for non-convex problems
    # Use a large population and maxiter to ensure convergence
    result = differential_evolution(
        _objective, 
        bounds, 
        maxiter=500, 
        popsize=50, 
        tol=1e-8, 
        seed=42,
        polish=True
    )
    
    return np.array(result.x)
# EVOLVE-BLOCK-END
