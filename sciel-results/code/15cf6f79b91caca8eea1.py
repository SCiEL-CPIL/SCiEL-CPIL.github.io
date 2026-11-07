import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize
    t_n = (t - 30.0) / 20.0
    x_n = x / 0.2
    v_n = v / 0.35
    
    # Centers
    centers = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, -1.0, 0.0],
        [0.5, 0.0, 1.0],
        [0.5, 0.0, -1.0],
        [0.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [0.5, 0.5, 0.5],
        [0.5, -0.5, -0.5]
    ])
    
    # Input vector shape (N, 3)
    Z = np.column_stack((t_n, x_n, v_n))
    
    # Compute squared distances
    # Z: (N, 3), centers: (10, 3)
    # dist_sq[i, j] = ||Z[i] - centers[j]||^2
    # Use broadcasting
    diff = Z[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    sigma = 0.5
    rbf = np.exp(-dist_sq / (2 * sigma**2))
    
    # Linear combination
    return rbf @ params
