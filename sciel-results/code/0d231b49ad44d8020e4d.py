import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Define 10 Gaussian RBF centers
    centers = [
        (-1.0, -1.0),
        (-1.0, 0.0),
        (-1.0, 1.0),
        (0.0, -1.0),
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, -1.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.5, 0.5)
    ]
    
    sigma = 0.7
    inv_2_sigma2 = 1.0 / (2.0 * sigma * sigma)
    
    # Pre-compute basis functions for efficiency
    # x and y are arrays of shape (N,)
    basis = np.empty((10, len(x)), dtype=float)
    
    for i, (cx, cy) in enumerate(centers):
        dx = x - cx
        dy = y - cy
        dist_sq = dx * dx + dy * dy
        basis[i] = np.exp(-dist_sq * inv_2_sigma2)
        
    # Linear combination
    # params is shape (10,)
    # result is shape (N,)
    return np.sum(params * basis, axis=0)
