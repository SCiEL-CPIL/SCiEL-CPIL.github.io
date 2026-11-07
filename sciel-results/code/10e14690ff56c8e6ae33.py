import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if p.shape != (10,):
        p = np.resize(p, 10)
        
    # Centers: 8 corners + 1 center
    centers = np.array([
        [0.5, 0.5, 0.5],
        [2.0, 0.5, 0.5],
        [0.5, 2.0, 0.5],
        [2.0, 2.0, 0.5],
        [0.5, 0.5, 2.0],
        [2.0, 0.5, 2.0],
        [0.5, 2.0, 2.0],
        [2.0, 2.0, 2.0],
        [1.25, 1.25, 1.25]
    ])
    
    # Input matrix
    X = np.column_stack((x, y, z))
    
    # Compute distances
    # X: (N, 3), Centers: (9, 3)
    # dist_sq[i, j] = ||X_i - C_j||^2
    X_sq = np.sum(X**2, axis=1)[:, np.newaxis]
    C_sq = np.sum(centers**2, axis=1)[np.newaxis, :]
    X_C = X @ centers.T
    dist_sq = X_sq + C_sq - 2 * X_C
    dist_sq = np.maximum(dist_sq, 0)
    
    # RBF
    sigma = 0.6
    B = np.exp(-dist_sq / (2 * sigma**2))
    
    # pred = p[0] + B @ p[1:]
    pred = p[0] + B @ p[1:10]
    
    return pred
