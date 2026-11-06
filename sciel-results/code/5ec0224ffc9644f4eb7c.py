import numpy as np

def construct():
    # Generate 8-point Gauss-Legendre nodes and weights on [-1, 1]
    x, w = np.polynomial.legendre.leggauss(8)
    
    # Map nodes from [-1, 1] to [0, 1]
    # z = (x + 1) / 2
    z_nodes = (x + 1.0) / 2.0
    
    # Scale weights by the derivative of the mapping (1/2)
    z_weights = w * 0.5
    
    # Ensure nodes are sorted in strictly increasing order
    # leggauss returns them in increasing order, but let's be safe
    sort_idx = np.argsort(z_nodes)
    z_nodes = z_nodes[sort_idx]
    z_weights = z_weights[sort_idx]
    
    # Verify constraints
    # Nodes in [0, 1]
    assert np.all(z_nodes >= 0.0) and np.all(z_nodes <= 1.0)
    # Weights positive
    assert np.all(z_weights > 0.0)
    # Sum to 1
    assert np.isclose(np.sum(z_weights), 1.0, atol=1e-12)
    
    return {"nodes": z_nodes, "weights": z_weights}
