import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled with a Radial Basis Function (RBF) interpolant.
    10 Gaussian basis functions with fixed centers:
    - 4 corners of the normalized hypercube
    - 4 mid-edge centers
    - 1 centroid
    - 1 additional center for flexibility
    
    params: weights for each RBF basis function
    """
    # Normalize inputs to [0, 1] based on training ranges
    b_n = (b - 1.0) / 4.0
    s_n = (s - 1.0) / 4.0
    T_n = (temp - 1.0) / 39.0
    p_n = (pH - 2.0) / 10.0
    
    # Clip to [0, 1] for stability
    b_n = np.clip(b_n, 0.0, 1.0)
    s_n = np.clip(s_n, 0.0, 1.0)
    T_n = np.clip(T_n, 0.0, 1.0)
    p_n = np.clip(p_n, 0.0, 1.0)
    
    # Define 10 RBF centers in 4D normalized space
    # Centers: [b, s, T, p]
    centers = np.array([
        [0.0, 0.0, 0.0, 0.0],  # corner 1
        [1.0, 1.0, 1.0, 1.0],  # corner 2
        [0.0, 1.0, 0.0, 1.0],  # corner 3
        [1.0, 0.0, 1.0, 0.0],  # corner 4
        [0.5, 0.0, 0.5, 0.0],  # mid-edge 1
        [0.0, 0.5, 0.0, 0.5],  # mid-edge 2
        [1.0, 0.5, 1.0, 0.5],  # mid-edge 3
        [0.5, 1.0, 0.5, 1.0],  # mid-edge 4
        [0.5, 0.5, 0.5, 0.5],  # centroid
        [0.3, 0.7, 0.5, 0.3],  # additional center for flexibility
    ])
    
    # Gaussian width (fixed)
    sigma = 0.35
    
    # Stack inputs into 4D
    x = np.stack([b_n, s_n, T_n, p_n], axis=-1)  # shape: (n, 4)
    
    # Compute distances to each center
    # x shape: (n, 1, 4), centers shape: (1, 10, 4)
    diffs = x[:, np.newaxis, :] - centers[np.newaxis, :, :]  # shape: (n, 10, 4)
    dist2 = np.sum(diffs ** 2, axis=-1)  # shape: (n, 10)
    
    # Compute RBF basis values
    basis = np.exp(-dist2 / (2.0 * sigma ** 2))  # shape: (n, 10)
    
    # Weighted sum of basis functions
    y = basis @ params  # shape: (n,)
    
    return y
