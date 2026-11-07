"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF (Radial Basis Function) interpolant for E. coli growth rate.
    Uses 10 fixed Gaussian centers in the normalized 4D input space.
    The 10 parameters are the weights for each RBF, plus a bias-like offset.
    This provides a flexible, smooth, and differentiable model for BFGS.
    """
    # Normalize inputs to [0, 1] range based on known training ranges
    b_norm = (b - 1.0) / 4.0       # [1, 4.998] -> [0, ~1]
    s_norm = (s - 1.0) / 39.0      # [1, 39.98] -> [0, ~1]
    temp_norm = (temp - 1.0) / 11.0  # [1, 12] -> [0, ~1]
    ph_norm = (pH - 1.0) / 4.0      # [1, 5] -> [0, ~1]
    
    # Define 10 fixed centers spread across the 4D normalized input space
    # Using a structured grid-like placement for good coverage
    centers = np.array([
        [0.1, 0.1, 0.1, 0.1],
        [0.9, 0.1, 0.1, 0.1],
        [0.1, 0.9, 0.1, 0.1],
        [0.9, 0.9, 0.1, 0.1],
        [0.1, 0.1, 0.9, 0.1],
        [0.9, 0.1, 0.9, 0.1],
        [0.1, 0.9, 0.9, 0.1],
        [0.9, 0.9, 0.9, 0.1],
        [0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.9],
    ])
    
    # RBF width: chosen to provide good local coverage in normalized space
    width = 0.45
    
    # Build the input matrix: (n_samples, 4)
    X = np.column_stack([b_norm, s_norm, temp_norm, ph_norm])
    
    # Compute squared distances from each sample to each center: (n_samples, 10)
    # Using broadcasting: X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dist = np.sum(diff ** 2, axis=2)  # (n_samples, 10)
    
    # Compute RBF values: exp(-sq_dist / (2 * width^2))
    rbf_vals = np.exp(-sq_dist / (2.0 * width ** 2))  # (n_samples, 10)
    
    # params: 10 weights for the 10 RBFs
    weights = params  # shape (10,)
    
    # Prediction: weighted sum of RBF values
    # rbf_vals: (n_samples, 10), weights: (10,) -> (n_samples,)
    prediction = rbf_vals @ weights
    
    return prediction
# EVOLVE-BLOCK-END
