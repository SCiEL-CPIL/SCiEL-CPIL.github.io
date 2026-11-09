import numpy as np

# Pre-defined centers for RBFs
_CENTERS = np.array([
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
_SIGMA = 0.8
_INV_2_SIGMA_SQ = 1.0 / (2.0 * _SIGMA * _SIGMA)

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # x, y, z are arrays of shape (N,)
    # params is shape (10,)
    
    # Compute squared distances to centers
    # x, y, z are 1D arrays. We need to broadcast.
    # Centers shape (9, 3). Inputs shape (N,).
    # dist_sq[i, j] = (x[j] - c_i[0])^2 + ...
    
    dx = x[:, np.newaxis] - _CENTERS[:, 0]
    dy = y[:, np.newaxis] - _CENTERS[:, 1]
    dz = z[:, np.newaxis] - _CENTERS[:, 2]
    
    dist_sq = dx**2 + dy**2 + dz**2
    
    # RBF values
    rbf_vals = np.exp(_INV_2_SIGMA_SQ * dist_sq) # Shape (N, 9)
    
    # Add constant term
    # Basis matrix shape (N, 10)
    basis = np.hstack((np.ones((x.shape[0], 1)), rbf_vals))
    
    # Prediction
    pred = basis @ params
    
    # Ensure finite
    if not np.all(np.isfinite(pred)):
        return np.zeros_like(x)
        
    return pred
