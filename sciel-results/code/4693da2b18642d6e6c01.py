"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard: ensure we return within 150s wall-clock
    t_start = time.time()
    deadline = 145.0  # 5s safety margin
    
    # Construct 10 basis functions using Radial Basis Functions (Gaussian kernels)
    # and a constant term to capture local and global features.
    # Centers are placed on a grid to cover the [-1, 1]^2 domain.
    
    # Define centers for 9 Gaussian RBFs
    centers = [
        (0.0, 0.0),
        (1.0, 1.0),
        (1.0, -1.0),
        (-1.0, 1.0),
        (-1.0, -1.0),
        (0.5, 0.5),
        (0.5, -0.5),
        (-0.5, 0.5),
        (-0.5, -0.5)
    ]
    
    # Width parameter for Gaussian kernels
    sigma = 1.0
    inv_2_sigma_sq = 1.0 / (2.0 * sigma * sigma)
    
    b0 = np.exp(-((x - centers[0][0])**2 + (y - centers[0][1])**2) * inv_2_sigma_sq)
    b1 = np.exp(-((x - centers[1][0])**2 + (y - centers[1][1])**2) * inv_2_sigma_sq)
    b2 = np.exp(-((x - centers[2][0])**2 + (y - centers[2][1])**2) * inv_2_sigma_sq)
    b3 = np.exp(-((x - centers[3][0])**2 + (y - centers[3][1])**2) * inv_2_sigma_sq)
    b4 = np.exp(-((x - centers[4][0])**2 + (y - centers[4][1])**2) * inv_2_sigma_sq)
    b5 = np.exp(-((x - centers[5][0])**2 + (y - centers[5][1])**2) * inv_2_sigma_sq)
    b6 = np.exp(-((x - centers[6][0])**2 + (y - centers[6][1])**2) * inv_2_sigma_sq)
    b7 = np.exp(-((x - centers[7][0])**2 + (y - centers[7][1])**2) * inv_2_sigma_sq)
    b8 = np.exp(-((x - centers[8][0])**2 + (y - centers[8][1])**2) * inv_2_sigma_sq)
    b9 = np.ones_like(x)  # Constant term (bias)
    
    pred = (params[0] * b0 + params[1] * b1 + params[2] * b2 + 
            params[3] * b3 + params[4] * b4 + params[5] * b5 + 
            params[6] * b6 + params[7] * b7 + params[8] * b8 + 
            params[9] * b9)
    
    # Check deadline before returning
    if time.time() - t_start > deadline:
        return pred
    
    return pred
# EVOLVE-BLOCK-END
