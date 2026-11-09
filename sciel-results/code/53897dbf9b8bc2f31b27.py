"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # RBF basis with 10 fixed centers evenly spaced in [0.5, 2]^3
    # Centers: a 2x2x2 grid plus 2 additional centers for better coverage
    # We use 10 centers total to match the 10 params
    
    # Generate 10 centers: 8 corners of [0.5,2]^3 plus 2 face centers
    cx = [0.5, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 2.0, 1.25, 1.25]
    cy = [0.5, 0.5, 2.0, 2.0, 0.5, 0.5, 2.0, 2.0, 0.5, 2.0]
    cz = [0.5, 2.0, 0.5, 2.0, 0.5, 2.0, 0.5, 2.0, 1.25, 1.25]
    
    centers = np.array(list(zip(cx, cy, cz)))
    
    # Fixed bandwidth gamma for the Gaussian kernel
    # Domain size is 1.5 in each dimension, so typical distance ~ 1-2
    # gamma controls width; we use a moderate value
    gamma = 2.0
    
    # Compute RBF features: exp(-gamma * ||(x,y,z) - center_i||^2)
    # x, y, z are arrays of shape (n,)
    points = np.stack([x, y, z], axis=-1)  # shape (n, 3)
    
    # Compute squared distances to each center
    # points: (n, 3), centers: (10, 3)
    diff = points[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (n, 10, 3)
    sq_dist = np.sum(diff ** 2, axis=2)  # (n, 10)
    
    # RBF features
    rbf = np.exp(-gamma * sq_dist)  # (n, 10)
    
    # Linear combination with params
    return rbf @ params
# EVOLVE-BLOCK-END
