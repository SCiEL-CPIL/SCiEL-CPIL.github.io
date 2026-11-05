"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Gauss-Legendre quadrature rule with 8 nodes on [0, 1]
    # This is a spectral initialization which is optimal for polynomials
    # and highly accurate for smooth functions like 1/E(zt).
    n = 8
    # Get nodes and weights on [-1, 1]
    nodes_gl, weights_gl = np.polynomial.legendre.leggauss(n)
    
    # Map nodes to [0, 1]
    nodes = (nodes_gl + 1.0) / 2.0
    
    # Map weights to [0, 1]
    weights = weights_gl / 2.0
    
    # Ensure strict increasing order (leggauss returns sorted, but just in case)
    sort_idx = np.argsort(nodes)
    nodes = nodes[sort_idx]
    weights = weights[sort_idx]
    
    # Verify constraints
    # Nodes in [0, 1]
    assert np.all(nodes > 0) and np.all(nodes < 1)
    # Weights positive
    assert np.all(weights > 0)
    # Sum to 1
    assert abs(np.sum(weights) - 1.0) < 1e-12
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
