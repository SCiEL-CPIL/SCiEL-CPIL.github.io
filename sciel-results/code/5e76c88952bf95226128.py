import numpy as np
import scipy.special

def construct():
    # Use Gauss-Jacobi quadrature with alpha=0, beta=1
    # This is optimal for functions behaving like (1+x)^1 near x=-1 (i.e., u^1 near u=0)
    # The integrand f(u) ~ u near u=0.
    
    # n=8 points
    x, w = scipy.special.roots_jacobi(8, 0, 1)
    
    # Map nodes from [-1, 1] to [0, 1]
    # u = (x + 1) / 2
    nodes = (x + 1.0) / 2.0
    
    # The Gauss-Jacobi rule approximates:
    # integral_{-1}^{1} (1-x)^0 (1+x)^1 f(x) dx = sum w_i f(x_i)
    # We want to approximate integral_{0}^{1} g(u) du.
    # Substitution u = (x+1)/2 => dx = 2 du.
    # integral_{0}^{1} g(u) du = integral_{-1}^{1} g((x+1)/2) * 0.5 dx
    # We identify f(x) = 0.5 * g((x+1)/2) / (1+x)
    # Note 1+x = 2u. So f(x) = g(u) / (4u).
    # The quadrature sum is sum w_i * f(x_i) = sum w_i * g(u_i) / (4 u_i).
    # Thus, the effective weights for g(u) are W_i = w_i / (4 u_i).
    
    # Calculate effective weights
    weights = w / (4.0 * nodes)
    
    # Normalize weights to sum to 1 exactly (to satisfy constraint)
    # The sum should theoretically be 1, but normalize for safety.
    weights = weights / np.sum(weights)
    
    # Ensure nodes are strictly increasing (they are from roots_jacobi)
    # Ensure weights are positive (they are)
    
    return {"nodes": nodes, "weights": weights}
