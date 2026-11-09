"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF-based symbolic regression model.
    
    Uses 10 fixed Gaussian radial basis functions centered in the input domain [0.5, 2]^3.
    The model is linear in the parameters, allowing BFGS to converge reliably.
    """
    # Ensure inputs are float arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Define 10 fixed centers in the domain [0.5, 2.0]
    # We distribute them roughly on a grid. 
    # 2 points in x, 2 in y, 2 in z gives 8. We can add 2 more or adjust.
    # Let's use a simple set of 10 points covering the space.
    # x_coords: 0.5, 1.25, 2.0
    # y_coords: 0.5, 1.25, 2.0
    # z_coords: 0.5, 1.25, 2.0
    # Total 27 possible. We pick 10 spread out.
    
    # Centers:
    # 1: (0.5, 0.5, 0.5)
    # 2: (2.0, 0.5, 0.5)
    # 3: (0.5, 2.0, 0.5)
    # 4: (2.0, 2.0, 0.5)
    # 5: (0.5, 0.5, 2.0)
    # 6: (2.0, 0.5, 2.0)
    # 7: (0.5, 2.0, 2.0)
    # 8: (2.0, 2.0, 2.0)
    # 9: (1.25, 1.25, 0.5)
    # 10: (1.25, 1.25, 2.0)
    
    centers = np.array([
        [0.5, 0.5, 0.5],
        [2.0, 0.5, 0.5],
        [0.5, 2.0, 0.5],
        [2.0, 2.0, 0.5],
        [0.5, 0.5, 2.0],
        [2.0, 0.5, 2.0],
        [0.5, 2.0, 2.0],
        [2.0, 2.0, 2.0],
        [1.25, 1.25, 0.5],
        [1.25, 1.25, 2.0]
    ])
    
    # Stack input points into (N, 3) array
    inputs = np.column_stack((x, y, z))
    
    # Compute squared distances from each input point to each center
    # inputs: (N, 3), centers: (10, 3)
    # dist_sq: (N, 10)
    diff = inputs[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    
    # Gaussian kernel: exp(-gamma * dist_sq)
    # Fixed bandwidth gamma. 
    # Domain size is 1.5. Max dist^2 is 3 * 1.5^2 = 6.75.
    # If gamma=1, min weight is exp(-6.75) ~ 0.001.
    # If gamma=0.5, min weight is exp(-3.375) ~ 0.034.
    # Let's use gamma=1.0 for a sharper local basis, which helps with non-smooth or complex functions.
    gamma = 1.0
    
    rbf_vals = np.exp(-gamma * dist_sq)
    
    # Prediction is linear combination of RBF values
    # rbf_vals: (N, 10), p: (10,)
    pred = rbf_vals @ p
    
    return pred
# EVOLVE-BLOCK-END
