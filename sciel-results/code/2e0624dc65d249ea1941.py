import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 60.0  # Safety margin
    
    # Gauss-Legendre 8-point nodes and weights on [-1, 1]
    # Using high precision values
    t = np.array([
        -0.9602898564975363,
        -0.7966664774136267,
        -0.5255324099163290,
        -0.1834346425004484,
         0.1834346425004484,
         0.5255324099163290,
         0.7966664774136267,
         0.9602898564975363
    ])
    w = np.array([
        0.1012285362903763,
        0.2223810344533745,
        0.3137066458778873,
        0.3626837833783620,
        0.3626837833783620,
        0.3137066458778873,
        0.2223810344533745,
        0.1012285362903763
    ])
    
    # Map to [0, 1]
    nodes = 0.5 * t + 0.5
    weights = 0.5 * w
    
    # Ensure weights sum to 1 (they should already, but normalize to be safe)
    weights = weights / np.sum(weights)
    
    # Check constraints
    if not np.all(np.diff(nodes) > 0) or not np.all(weights > 0) or abs(np.sum(weights) - 1.0) > 1e-12:
        # Fallback to midpoint if something went wrong (should not happen)
        nodes = (np.arange(8) + 0.5) / 8.0
        weights = np.full(8, 1.0 / 8.0)
        
    return {"nodes": nodes, "weights": weights}
