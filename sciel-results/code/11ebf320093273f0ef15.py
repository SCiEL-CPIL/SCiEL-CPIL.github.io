"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Multi-scale sparse linear combination of 10 Radial Basis Functions (RBFs).
    
    Uses 5 wide RBFs and 5 narrow RBFs with centers from a Halton sequence.
    The optimizer fits the 10 amplitudes (params).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    params = np.asarray(params).ravel()
    
    # Helper to generate Halton sequence values
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f = f / base
            r = r + f * (index % base)
            index = index // base
        return r

    # Generate 10 Halton points in [0,1]^2, scale to [-1,1]^2
    centers = np.zeros((10, 2))
    for i in range(10):
        idx = i + 1
        centers[i, 0] = 2.0 * halton(idx, 2) - 1.0
        centers[i, 1] = 2.0 * halton(idx, 3) - 1.0

    # Interleave centers for wide and narrow scales to ensure spatial coverage
    c_wide = centers[0::2]    # Indices 0, 2, 4, 6, 8
    c_narrow = centers[1::2]  # Indices 1, 3, 5, 7, 9
    
    # Define widths
    sigma_wide = 0.8
    sigma_narrow = 0.4
    
    # Compute wide RBFs
    dx_w = x[:, np.newaxis] - c_wide[:, 0][np.newaxis, :]
    dy_w = y[:, np.newaxis] - c_wide[:, 1][np.newaxis, :]
    dist_w_sq = dx_w**2 + dy_w**2
    rbf_wide = np.exp(-dist_w_sq / (2.0 * sigma_wide**2))
    
    # Compute narrow RBFs
    dx_n = x[:, np.newaxis] - c_narrow[:, 0][np.newaxis, :]
    dy_n = y[:, np.newaxis] - c_narrow[:, 1][np.newaxis, :]
    dist_n_sq = dx_n**2 + dy_n**2
    rbf_narrow = np.exp(-dist_n_sq / (2.0 * sigma_narrow**2))
    
    # Combine predictions
    pred_wide = rbf_wide @ params[:5]
    pred_narrow = rbf_narrow @ params[5:]
    
    return pred_wide + pred_narrow
# EVOLVE-BLOCK-END
