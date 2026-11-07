import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using RBF (Radial Basis Function) interpolant.
    
    Inputs are normalized to [0,1] range based on training ranges:
    b: [1, 4.998] -> b_norm = (b-1)/3.998
    s: [1, 4.999] -> s_norm = (s-1)/3.999
    temp: [1.002, 39.98] -> temp_norm = (temp-1.002)/38.978
    pH: [2, 12] -> pH_norm = (pH-2)/10
    
    The model uses 10 Gaussian RBF centers:
    - 4 corner centers of the hypercube
    - 4 mid-edge centers (pairwise)
    - 1 centroid
    - 1 additional center at a strategic location
    
    The 10 parameters are the weights for these basis functions.
    """
    # Normalize inputs to [0, 1]
    b_norm = (b - 1.0) / 3.998
    s_norm = (s - 1.0) / 3.999
    temp_norm = (temp - 1.002) / 38.978
    pH_norm = (pH - 2.0) / 10.0
    
    # Ensure inputs are within [0,1] for stable RBF evaluation
    b_norm = np.clip(b_norm, 0.0, 1.0)
    s_norm = np.clip(s_norm, 0.0, 1.0)
    temp_norm = np.clip(temp_norm, 0.0, 1.0)
    pH_norm = np.clip(pH_norm, 0.0, 1.0)
    
    n = b.shape[0]
    
    # Define 10 RBF centers in 4D space
    # Corners (4): [0,0,0,0], [1,0,0,0], [0,1,0,0], [0,0,1,0]
    # Mid-edges (4): [0.5,0.5,0,0], [0,0.5,0.5,0], [0,0,0.5,0.5], [0.5,0,0.5,0.5]
    # Centroid (1): [0.5, 0.5, 0.5, 0.5]
    # Additional (1): [0.25, 0.25, 0.25, 0.25]
    
    centers = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.5, 0.5, 0.0],
        [0.0, 0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5],
        [0.25, 0.25, 0.25, 0.25]
    ])
    
    # Build input matrix (n x 4)
    X = np.column_stack([b_norm, s_norm, temp_norm, pH_norm])
    
    # Compute squared distances from each input to each center
    # X: (n, 4), centers: (10, 4)
    # dist_sq[i, j] = ||X[i] - centers[j]||^2
    
    # Efficient computation using broadcasting
    # X[:, np.newaxis, :] has shape (n, 1, 4)
    # centers[np.newaxis, :, :] has shape (1, 10, 4)
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (n, 10, 4)
    dist_sq = np.sum(diff ** 2, axis=2)  # (n, 10)
    
    # Gaussian RBF: exp(-dist_sq / (2 * sigma^2))
    # Use a fixed bandwidth sigma for all centers
    sigma_sq = 0.25  # sigma = 0.5, so 2*sigma^2 = 0.5
    rbf_values = np.exp(-dist_sq / 0.5)  # (n, 10)
    
    # Prediction: weighted sum of RBF values
    pred = rbf_values @ params
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
