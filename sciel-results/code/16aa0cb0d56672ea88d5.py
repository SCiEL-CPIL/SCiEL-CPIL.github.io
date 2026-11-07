"""Symbolic regression for E. coli growth rate - RBF interpolant with 10 Gaussian basis functions."""
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF (Radial Basis Function) model for E. coli growth rate with 10 parameters.
    Uses 10 Gaussian basis functions centered at strategic points in the normalized
    input space. The parameters are the weights for each basis function.
    
    Centers (in normalized [0,1]^4 space):
        c0: [0, 0, 0, 0]     - corner
        c1: [1, 1, 1, 1]     - corner
        c2: [0.5, 0.5, 0.5, 0.5] - centroid
        c3: [0, 1, 0, 1]     - corner
        c4: [1, 0, 1, 0]     - corner
        c5: [0, 0, 1, 1]     - corner
        c6: [1, 1, 0, 0]     - corner
        c7: [0.5, 0, 0.5, 0] - mid-edge
        c8: [0.5, 1, 0.5, 1] - mid-edge
        c9: [0, 0.5, 0, 0.5] - mid-edge
    
    Parameters:
        p[0..9]: weights for the 10 Gaussian basis functions
    """
    p = params
    
    # Normalize inputs to [0, 1] range
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 39.0
    temp_norm = (temp - 1.0) / 11.0
    ph_norm = (pH - 1.0) / 4.0
    
    # Clip normalized inputs to [0, 1] for stability
    b_norm = np.clip(b_norm, 0.0, 1.0)
    s_norm = np.clip(s_norm, 0.0, 1.0)
    temp_norm = np.clip(temp_norm, 0.0, 1.0)
    ph_norm = np.clip(ph_norm, 0.0, 1.0)
    
    # Define 10 center points in normalized space
    # Corners and strategic mid-points
    centers = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5, 0.5],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.5, 0.0, 0.5, 0.0],
        [0.5, 1.0, 0.5, 1.0],
        [0.0, 0.5, 0.0, 0.5]
    ])
    
    # Stack normalized inputs into matrix of shape (n_samples, 4)
    inputs = np.stack([b_norm, s_norm, temp_norm, ph_norm], axis=-1)
    
    # Compute squared distances from each input to each center
    # inputs: (n, 4), centers: (10, 4)
    # dist_sq: (n, 10)
    dist_sq = np.sum((inputs[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=-1)
    
    # Gaussian RBF with bandwidth sigma=0.5 (covers the [0,1] range well)
    sigma = 0.5
    rbf_values = np.exp(-dist_sq / (2.0 * sigma ** 2))
    
    # Compute weighted sum of RBF values
    # rbf_values: (n, 10), p: (10,)
    mu = np.sum(rbf_values * p[np.newaxis, :], axis=-1)
    
    # Clip to avoid extreme values
    mu = np.clip(mu, -10.0, 10.0)
    
    return mu
