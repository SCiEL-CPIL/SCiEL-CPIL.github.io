# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # RBF basis with 9 fixed centers (8 corners + 1 center)
    # 9 coefficients + 1 log-gamma parameter = 10 params
    
    # Centers: 8 corners of [0.5, 2]^3 plus the center point
    cx = [0.5, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 2.0, 1.25]
    cy = [0.5, 0.5, 2.0, 2.0, 0.5, 0.5, 2.0, 2.0, 1.25]
    cz = [0.5, 2.0, 0.5, 2.0, 0.5, 2.0, 0.5, 2.0, 1.25]
    
    centers = np.array(list(zip(cx, cy, cz)))  # shape (9, 3)
    
    # Extract coefficients and log-gamma
    coeffs = params[:9]
    log_gamma = params[9]
    
    # Clip log_gamma to avoid extreme values (e.g., gamma > 100 or < 0.01)
    # exp(10) ~ 22000, exp(-10) ~ 0.00004
    log_gamma = np.clip(log_gamma, -10, 10)
    gamma = np.exp(log_gamma)
    
    # Compute RBF features
    points = np.stack([x, y, z], axis=-1)  # shape (n, 3)
    
    # Compute squared distances to each center
    # points: (n, 3), centers: (9, 3)
    diff = points[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (n, 9, 3)
    sq_dist = np.sum(diff ** 2, axis=2)  # (n, 9)
    
    # RBF features
    rbf = np.exp(-gamma * sq_dist)  # (n, 9)
    
    # Linear combination with params
    return rbf @ coeffs
# EVOLVE-BLOCK-END
