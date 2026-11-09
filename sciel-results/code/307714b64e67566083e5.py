"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Radial Basis Function (RBF) model with 5 fixed centers
    # Centers evenly spaced across [-3, 3]
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    # params[0:5] are weights for each RBF
    # params[5:10] are width parameters for each RBF
    weights = params[0:5]
    widths = params[5:10]
    
    # Compute distances from each x to each center
    # x shape: (n,), centers shape: (5,)
    # diff shape: (n, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Compute squared distances
    sq_dist = diff ** 2
    
    # Compute RBF values: exp(-width * sq_dist)
    # width shape: (5,), sq_dist shape: (n, 5)
    # rbf shape: (n, 5)
    rbf = np.exp(-widths[np.newaxis, :] * sq_dist)
    
    # Combine with weights
    y = rbf @ weights
    
    return y
# EVOLVE-BLOCK-END
