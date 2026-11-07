"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using an RBF (Radial Basis Function) interpolant.
    
    Inputs:
        b: population density (normalized to [0, 1])
        s: substrate concentration (normalized to [0, 1])
        temp: temperature (normalized to [0, 1])
        pH: pH level (normalized to [0, 1])
        params: array of 10 fitted constants (weights for RBF centers + 1 bias)
    
    Returns:
        Predicted growth rate for each row.
    """
    # Normalize inputs to [0, 1] based on training ranges
    b_norm = (b - 1.0) / (4.998 - 1.0)
    s_norm = (s - 1.0) / (4.999 - 1.0)
    temp_norm = (temp - 1.002) / (39.98 - 1.002)
    pH_norm = (pH - 2.0) / (12.0 - 2.0)
    
    # Clip to [0, 1] for safety
    b_norm = np.clip(b_norm, 0, 1)
    s_norm = np.clip(s_norm, 0, 1)
    temp_norm = np.clip(temp_norm, 0, 1)
    pH_norm = np.clip(pH_norm, 0, 1)
    
    # 9 RBF centers: 8 corners of the hypercube + centroid
    # This gives good coverage of the input space
    centers = np.array([
        [0.0, 0.0, 0.0, 0.0],  # Corner 0
        [1.0, 0.0, 0.0, 0.0],  # Corner 1
        [0.0, 1.0, 0.0, 0.0],  # Corner 2
        [0.0, 0.0, 1.0, 0.0],  # Corner 3
        [0.0, 0.0, 0.0, 1.0],  # Corner 4
        [1.0, 1.0, 0.0, 0.0],  # Corner 5
        [1.0, 0.0, 1.0, 0.0],  # Corner 6
        [0.0, 1.0, 1.0, 0.0],  # Corner 7
        [0.5, 0.5, 0.5, 0.5],  # Centroid
    ])
    
    # Stack normalized inputs
    X = np.column_stack([b_norm, s_norm, temp_norm, pH_norm])
    
    # Compute squared distances from each sample to each center
    # X: (n_samples, 4), centers: (9, 4)
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dist = np.sum(diff ** 2, axis=2)  # (n_samples, 9)
    
    # Gaussian RBF with gamma tuned for good conditioning
    gamma = 2.0
    rbf_values = np.exp(-gamma * sq_dist)  # (n_samples, 9)
    
    # params[0:9] are RBF weights, params[9] is bias
    prediction = rbf_values @ params[:9] + params[9]
    
    return prediction
# EVOLVE-BLOCK-END
