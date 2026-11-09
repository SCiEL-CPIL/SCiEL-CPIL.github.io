"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from numpy.polynomial.legendre import leggauss, legder, legval

def construct():
    N = 8
    # Gauss-Lobatto nodes and weights on [-1, 1]
    # Nodes: -1, 1, and roots of P_{N-2}'(x)
    # Weights: w_i = 2 / (N(N-1) P_{N-1}(x_i)^2)
    
    # Get Legendre polynomials coefficients
    # We need P_{N-1}(x) and roots of P_{N-2}'(x)
    
    # Use leggauss to get P_{N-1} roots? No, we need specific nodes.
    # Let's construct P_{N-1} and P_{N-2}
    
    # Coefficients of Legendre polynomials
    # P_0 = 1
    # P_1 = x
    # P_n = ((2n-1)x P_{n-1} - (n-1) P_{n-2}) / n
    
    def legendre_coeffs(n):
        if n == 0:
            return np.array([1.0])
        if n == 1:
            return np.array([0.0, 1.0])
        
        c_prev2 = np.array([1.0]) # P_0
        c_prev1 = np.array([0.0, 1.0]) # P_1
        for k in range(2, n + 1):
            # (2k-1) x P_{k-1}
            term1 = (2 * k - 1) * np.convolve(c_prev1, [0.0, 1.0])
            # (k-1) P_{k-2}
            term2 = (k - 1) * c_prev2
            # P_k = (term1 - term2) / k
            c_curr = (term1 - term2) / k
            c_prev2 = c_prev1
            c_prev1 = c_curr
        return c_prev1

    c_N_minus_1 = legendre_coeffs(N - 1) # P_7
    c_N_minus_2 = legendre_coeffs(N - 2) # P_6
    
    # Derivative of P_{N-2}
    # d/dx P_{N-2}
    # If P(x) = sum a_i x^i, P'(x) = sum i a_i x^{i-1}
    d_c_N_minus_2 = np.arange(1, len(c_N_minus_2)) * c_N_minus_2[1:]
    # Pad with 0 if needed? No, degree drops by 1.
    
    # Find roots of P_{N-2}'(x)
    # Use np.roots
    interior_nodes = np.roots(d_c_N_minus_2)
    # Filter real roots
    interior_nodes = interior_nodes[np.abs(interior_nodes.imag) < 1e-10].real
    # Sort
    interior_nodes = np.sort(interior_nodes)
    
    # Full nodes on [-1, 1]
    x_nodes = np.concatenate(([-1.0], interior_nodes, [1.0]))
    
    # Calculate weights
    # w_i = 2 / (N(N-1) P_{N-1}(x_i)^2)
    weights = np.zeros(N)
    for i, x in enumerate(x_nodes):
        p_val = np.polyval(c_N_minus_1[::-1], x) # Coeffs are low to high, polyval is high to low
        weights[i] = 2.0 / (N * (N - 1) * p_val**2)
        
    # Map to [0, 1]
    # t = (x+1)/2
    # dt = dx/2
    nodes = (x_nodes + 1.0) / 2.0
    weights = weights / 2.0
    
    # Ensure strictly increasing
    # x_nodes are -1, ..., 1. Sorted.
    # nodes will be 0, ..., 1.
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
