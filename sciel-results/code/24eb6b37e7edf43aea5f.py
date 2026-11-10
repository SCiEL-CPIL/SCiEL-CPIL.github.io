import numpy as np
import time

def construct():
    """
    Eight-node positive quadrature rule for dimensionless luminosity distance.
    Uses Gauss-Legendre quadrature, which is optimal for smooth integrands
    with positive weights summing to 1.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5-second safety margin
    
    # Gauss-Legendre nodes and weights on [0, 1]
    x, w = np.polynomial.legendre.leggauss(8)
    nodes = 0.5 * (x + 1.0)
    weights = 0.5 * w
    
    # Normalize weights to sum to exactly 1
    weights = weights / np.sum(weights)
    
    # Verify constraints
    assert len(nodes) == 8
    assert len(weights) == 8
    assert np.all(np.diff(nodes) > 0), "Nodes must be strictly increasing"
    assert np.all(nodes >= 0) and np.all(nodes <= 1), "Nodes must be in [0,1]"
    assert np.all(weights > 0), "Weights must be positive"
    assert abs(np.sum(weights) - 1.0) < 1e-12, "Weights must sum to 1"
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Should not happen, but safety guard
        pass
    
    return {"nodes": nodes, "weights": weights}
