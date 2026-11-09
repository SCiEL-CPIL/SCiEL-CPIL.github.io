import numpy as np
from scipy.special import roots_laguerre, roots_genlaguerre, eval_laguerre, eval_genlaguerre
from scipy.optimize import minimize_scalar, brentq

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the 12 positive double-root locations for the refined
    Cohn-Goncalves Laguerre-polynomial construction (m=12, n=1).
    
    The 'refined' construction uses a linear combination of Laguerre polynomials
    to minimize the Fourier sign uncertainty upper bound. We search over the
    generalized Laguerre parameter alpha to find the optimal root configuration.
    """
    m = 12
    n = 1
    
    def upper_bound_for_roots(roots):
        """
        Compute the upper bound on C4 given a set of roots.
        The bound is related to the integral of the square of the polynomial
        formed by the roots, weighted appropriately.
        
        For a polynomial P(x) = prod (x - r_i), the relevant quantity for
        the uncertainty bound involves the L2 norm under the Laguerre weight.
        """
        # P(x) = prod_{i=1}^{m} (x - r_i)
        # The bound on C4 is proportional to the integral of |P(x)|^2 x^alpha e^{-x} dx
        # normalized by the integral of x^alpha e^{-x} dx (which is alpha! for integer alpha)
        
        # We'll compute the integral numerically using Gauss-Laguerre quadrature
        # with a high order for accuracy
        x_nodes, x_weights = roots_laguerre(200)
        
        # Evaluate P(x) at the quadrature nodes
        P_vals = np.ones_like(x_nodes)
        for r in roots:
            P_vals *= (x_nodes - r)
        
        # Integrate |P(x)|^2 x^alpha e^{-x} dx
        # The quadrature weights already include e^{-x}
        integral = np.sum(x_weights * (x_nodes ** alpha) * (P_vals ** 2))
        
        # Normalize by the norm of the weight function
        # integral of x^alpha e^{-x} dx = Gamma(alpha+1)
        norm = np.math.gamma(alpha + 1)
        
        return integral / norm
    
    def objective(alpha):
        """
        Objective function: compute the upper bound for generalized Laguerre roots
        with parameter alpha.
        """
        roots, _ = roots_genlaguerre(m, alpha)
        # Ensure all roots are positive (they should be for alpha > -1)
        if np.any(roots <= 0):
            return np.inf
        return upper_bound_for_roots(roots)
    
    # Search over alpha in a reasonable range
    # Standard Laguerre is alpha=0. The "refined" version might use alpha near 1 or other values.
    # Let's search alpha in [-0.5, 2.0]
    result = minimize_scalar(objective, bounds=(-0.5, 2.0), method='bounded', 
                             options={'xatol': 1e-8})
    
    optimal_alpha = result.x
    roots, _ = roots_genlaguerre(m, optimal_alpha)
    
    # Sort the roots for consistency
    roots = np.sort(roots)
    
    # Return as a list of floats
    return [float(r) for r in roots]
# EVOLVE-BLOCK-END
