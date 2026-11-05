"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares
import sys

def construct():
    # We design a quadrature rule by moment matching against a basis
    # that captures the typical shapes of the cosmological integrand 1/E(zx).
    # The integrand behaves like a constant for small z/flat DE, and like
    # (1+zx)^(-1.5) for large z/matter dominated.
    # We match moments of polynomials (for smoothness) and inverse powers
    # (for decay behavior).
    
    n_nodes = 8
    
    # Basis functions:
    # 1. Polynomials 1, x, ..., x^7
    # 2. Inverse powers 1/(1+x), 1/(1+x)^2, ..., 1/(1+x)^8
    # Total 16 moments to match.
    
    def basis_functions(x):
        # x is an array of nodes
        # Returns a matrix of shape (16, n_nodes)
        # Row 0: 1
        # Row 1: x
        # ...
        # Row 7: x^7
        # Row 8: 1/(1+x)
        # ...
        # Row 15: 1/(1+x)^8
        B = np.zeros((16, len(x)))
        for i in range(8):
            B[i, :] = x**i
        for i in range(8):
            B[8+i, :] = 1.0 / (1.0 + x)**(i+1)
        return B

    def moments():
        # Compute exact integrals of the basis functions over [0, 1]
        # int_0^1 x^k dx = 1/(k+1)
        # int_0^1 (1+x)^(-m) dx = [ (1+x)^(-m+1) / (-m+1) ]_0^1 for m != 1
        # int_0^1 1/(1+x) dx = ln(2)
        M = np.zeros(16)
        for k in range(8):
            M[k] = 1.0 / (k + 1.0)
        for m in range(1, 9):
            if m == 1:
                M[7 + m] = np.log(2.0)
            else:
                # integral of (1+x)^(-m) from 0 to 1
                # = (1/(-m+1)) * (2^(-m+1) - 1^(-m+1))
                M[7 + m] = (2.0**(-m+1) - 1.0) / (-m + 1.0)
        return M

    target_moments = moments()
    
    # Initial guess: Gauss-Legendre nodes and weights on [0, 1]
    # Generate GL nodes and weights on [-1, 1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(n_nodes)
    x0 = 0.5 * (x_gl + 1.0)
    w0 = 0.5 * w_gl
    
    # Parameters to optimize: nodes and weights
    # To ensure nodes are strictly increasing and in [0,1], we can parameterize
    # them using a transformation, or just constrain them.
    # Let's use least_squares with bounds.
    
    def residuals(params):
        nodes = params[:8]
        weights = params[8:]
        
        # Check constraints
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            return np.ones(16) * 1e6
        if np.any(np.diff(nodes) <= 0):
            return np.ones(16) * 1e6
        if np.any(weights <= 0):
            return np.ones(16) * 1e6
            
        B = basis_functions(nodes)
        # Sum w_i * B_j(x_i)
        calc_moments = B.T @ weights
        return calc_moments - target_moments

    # Initial guess
    p0 = np.concatenate([x0, w0])
    
    # Bounds
    lb_nodes = np.full(8, 1e-6)
    ub_nodes = np.full(8, 1 - 1e-6)
    lb_weights = np.full(8, 1e-12)
    ub_weights = np.full(8, 1.0)
    
    bounds = (np.concatenate([lb_nodes, lb_weights]),
              np.concatenate([ub_nodes, ub_weights]))
    
    try:
        res = least_squares(residuals, p0, bounds=bounds, max_nfev=2000)
        nodes = res.x[:8]
        weights = res.x[8:]
        
        # Sort nodes and corresponding weights if needed (though constrained to be increasing)
        # The optimizer might shuffle them if not constrained strictly, but we used diff check.
        # It's safer to sort.
        sort_idx = np.argsort(nodes)
        nodes = nodes[sort_idx]
        weights = weights[sort_idx]
        
        # Normalize weights to sum to 1 exactly (helps with floating point errors)
        sum_w = np.sum(weights)
        if sum_w > 0:
            weights = weights / sum_w
            
        # Final validation
        if not np.all(np.diff(nodes) > 0):
            raise ValueError("Nodes not strictly increasing")
        if not np.all(weights > 0):
            raise ValueError("Weights not positive")
        if not np.all(nodes > 0) or not np.all(nodes < 1):
            raise ValueError("Nodes out of bounds")
            
        return {"nodes": nodes, "weights": weights}
        
    except Exception as e:
        # Fallback to Gauss-Legendre if optimization fails
        return {"nodes": x0, "weights": w0}

# EVOLVE-BLOCK-END
