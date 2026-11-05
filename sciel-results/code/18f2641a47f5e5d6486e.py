# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Chebyshev-Gauss quadrature rule with 8 nodes on [0, 1].
    # This rule uses nodes clustered near the endpoints, which is beneficial
    # for cosmological integrands that often exhibit higher curvature or
    # steeper gradients near z=0 (observer) compared to the center of the
    # integration domain. The weights are uniform, simplifying the rule
    # while maintaining high accuracy for smooth functions.
    
    N = 8
    # Generate Chebyshev-Gauss nodes on [-1, 1]
    k = np.arange(N)
    x = np.cos((2 * k + 1) * np.pi / (2 * N))
    
    # Map nodes from [-1, 1] to [0, 1]
    nodes = 0.5 * (x + 1.0)
    
    # Ensure nodes are strictly increasing
    nodes = np.sort(nodes)
    
    # Chebyshev-Gauss weights are all equal on [-1, 1].
    # After mapping to [0, 1] and normalizing to sum to 1,
    # the weights are simply 1/N.
    weights = np.ones(N) / N
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
