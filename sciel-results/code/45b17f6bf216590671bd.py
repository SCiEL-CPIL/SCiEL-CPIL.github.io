import numpy as np

def construct():
    n = 8
    # Get Gauss-Legendre nodes and weights on [-1, 1]
    x, w = np.polynomial.legendre.leggauss(n)
    
    # Map to [0, 1]
    nodes = 0.5 * (x + 1.0)
    weights = 0.5 * w
    
    # Optional: Clean up numerical noise if necessary, but leggauss is precise.
    # Ensure sum is exactly 1?
    # s = np.sum(weights)
    # weights /= s
    # This might break the optimal property slightly if s is not exactly 1, but s is 1.0 within fp error.
    # The requirement is "within 1e-12".
    
    return {"nodes": nodes, "weights": weights}
