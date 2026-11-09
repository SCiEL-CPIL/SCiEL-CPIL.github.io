import numpy as np
from numpy.polynomial.legendre import Legendre

def construct():
    N = 8
    # Gauss-Lobatto-Legendre nodes and weights
    
    # P_{N-1}(x)
    P = Legendre.basis(N - 1)
    dP = P.deriv()
    
    # Roots of dP
    # dP is a Legendre series
    # Find roots
    # Legendre.roots() works on the object
    roots_interior = dP.roots()
    
    # Nodes in [-1, 1]
    nodes_x = np.concatenate(([-1.0], roots_interior, [1.0]))
    nodes_x = np.sort(nodes_x)
    
    # Map to [0, 1]
    nodes = (nodes_x + 1.0) / 2.0
    
    # Weights
    # w_i = 2 / (N(N-1) P(x_i)^2)
    weights_x = np.zeros(N)
    for i in range(N):
        pi_val = P(nodes_x[i])
        weights_x[i] = 2.0 / (N * (N - 1) * pi_val**2)
        
    # Map to [0, 1]
    weights = weights_x / 2.0
    
    # Normalize
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
