"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF model with Gaussian kernels centered at the four corners and the origin.
    params: length-10 array.
      params[0..4]: weights for the 5 RBF centers.
      params[5..9]: log-sigmas for the 5 RBF centers (exponentiated to get positive sigmas).
    """
    # Centers: 4 corners + origin
    cx = np.array([-1.0, 1.0, -1.0, 1.0, 0.0])
    cy = np.array([-1.0, -1.0, 1.0, 1.0, 0.0])
    
    # Extract weights and log-sigmas
    weights = params[0:5]
    log_sigmas = params[5:10]
    
    # Convert log-sigmas to positive sigmas
    # Adding a small epsilon to avoid zero division or extreme values if needed, 
    # but exp is always > 0.
    sigmas = np.exp(log_sigmas)
    
    # Compute squared distances from each input point to each center
    # x, y are arrays of shape (n,)
    # dist_sq[i, j] = (x[i] - cx[j])^2 + (y[i] - cy[j])^2
    dx = x[:, np.newaxis] - cx[np.newaxis, :]  # (n, 5)
    dy = y[:, np.newaxis] - cy[np.newaxis, :]  # (n, 5)
    dist_sq = dx ** 2 + dy ** 2  # (n, 5)
    
    # Compute Gaussian RBF values for each center
    # rbf_values[i, j] = exp(-dist_sq[i, j] / (2 * sigma_j^2))
    # We need to broadcast sigmas correctly: sigmas is shape (5,), dist_sq is (n, 5)
    # So we divide dist_sq by (2 * sigmas^2)
    inv_2_sigma_sq = 1.0 / (2.0 * sigmas ** 2)  # shape (5,)
    rbf_values = np.exp(-dist_sq * inv_2_sigma_sq[np.newaxis, :])  # (n, 5)
    
    # Prediction: sum of weighted RBFs
    pred = rbf_values @ weights
    
    return pred
# EVOLVE-BLOCK-END
