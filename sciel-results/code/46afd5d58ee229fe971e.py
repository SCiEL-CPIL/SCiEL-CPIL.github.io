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
    b_norm = (b - 1.0) / (4.999 - 1.0)
    s_norm = (s - 1.0) / (39.98 - 1.0)
    temp_norm = (temp - 1.0) / (12.0 - 1.0)
    pH_norm = (pH - 1.0) / (4.998 - 1.0)  # Assuming pH range similar to b, but let's use [1, 4.998] as given
    
    # Actually, looking at the input ranges:
    # b: [1, 4.998]
    # s: [1, 4.999]
    # temp: [1.002, 39.98]
    # pH: [2, 12]
    
    b_norm = (b - 1.0) / (4.998 - 1.0)
    s_norm = (s - 1.0) / (4.999 - 1.0)
    temp_norm = (temp - 1.002) / (39.98 - 1.002)
    pH_norm = (pH - 2.0) / (12.0 - 2.0)
    
    # Clip to [0, 1] for safety
    b_norm = np.clip(b_norm, 0, 1)
    s_norm = np.clip(s_norm, 0, 1)
    temp_norm = np.clip(temp_norm, 0, 1)
    pH_norm = np.clip(pH_norm, 0, 1)
    
    # Define 10 RBF centers in the normalized 4D space
    # 4 corners: [0,0,0,0], [1,1,1,1], [0,1,0,1], [1,0,1,0]
    # 4 mid-edges or centroid: [0.5,0.5,0.5,0.5], [0,0,0.5,0.5], [0.5,0.5,0,0], [1,1,0.5,0.5]
    # Let's use:
    # c0: [0, 0, 0, 0]
    # c1: [1, 1, 1, 1]
    # c2: [0, 1, 0, 1]
    # c3: [1, 0, 1, 0]
    # c4: [0.5, 0.5, 0.5, 0.5]  (centroid)
    # c5: [0, 0, 0.5, 0.5]
    # c6: [0.5, 0.5, 0, 0]
    # c7: [1, 1, 0.5, 0.5]
    # c8: [0, 0.5, 1, 0.5]
    # c9: [0.5, 0, 0.5, 1]
    
    centers = np.array([
        [0.5, 0.5, 0.5, 0.5], # Center
        [0.0, 0.5, 0.5, 0.5], # b min
        [1.0, 0.5, 0.5, 0.5], # b max
        [0.5, 0.0, 0.5, 0.5], # s min
        [0.5, 1.0, 0.5, 0.5], # s max
        [0.5, 0.5, 0.0, 0.5], # temp min
        [0.5, 0.5, 1.0, 0.5], # temp max
        [0.5, 0.5, 0.5, 0.0], # pH min
        [0.5, 0.5, 0.5, 1.0], # pH max
        [0.0, 0.0, 0.0, 0.0]  # Corner (low all)
    ])
    
    # Number of samples
    n_samples = b_norm.shape[0]
    
    # Stack normalized inputs
    X = np.column_stack([b_norm, s_norm, temp_norm, pH_norm])
    
    # Compute squared distances from each sample to each center
    # X: (n_samples, 4), centers: (10, 4)
    # diff: (n_samples, 10, 4)
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dist = np.sum(diff ** 2, axis=2)  # (n_samples, 10)
    
    # Gaussian RBF: exp(-gamma * sq_dist)
    # Use gamma = 1.0 for simplicity (could be tuned, but we only have 10 params as weights)
    gamma = 1.0
    rbf_values = np.exp(-gamma * sq_dist)  # (n_samples, 10)
    
    # params: (10,) weights for each RBF
    # Prediction: sum of weight_i * rbf_i
    prediction = rbf_values @ params[:10]
    
    return prediction
# EVOLVE-BLOCK-END
