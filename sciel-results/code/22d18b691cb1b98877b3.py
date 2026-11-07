"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian process-like kernel expansion basis using 10 fixed radial basis functions (RBFs).
    Centers: 3x3 grid in [-1, 1] plus the origin.
    params: length 10 array of coefficients for each RBF.
    """
    # Ensure x and y are 1D arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Define 10 centers: 3x3 grid + origin
    # Grid points: -1, 0, 1 for both x and y
    grid_vals = np.array([-1.0, 0.0, 1.0])
    
    # Generate 3x3 grid centers
    gx, gy = np.meshgrid(grid_vals, grid_vals)
    centers = np.column_stack((gx.ravel(), gy.ravel()))
    
    # Add origin (0,0) as the 10th center
    origin = np.array([[0.0, 0.0]])
    centers = np.vstack((centers, origin))
    
    # Now we have 10 centers
    # Compute distances from each input point to each center
    # x: (N,), y: (N,)
    # centers: (10, 2)
    
    # Reshape for broadcasting
    x_col = x[:, np.newaxis]  # (N, 1)
    y_col = y[:, np.newaxis]  # (N, 1)
    
    # centers: (10, 2) -> (1, 10, 2) for broadcasting with (N, 1, 2)
    cx = centers[:, 0][np.newaxis, :, np.newaxis]  # (1, 10, 1)
    cy = centers[:, 1][np.newaxis, :, np.newaxis]  # (1, 10, 1)
    
    # Compute squared distances: (N, 10)
    dist_sq = (x_col - cx) ** 2 + (y_col - cy) ** 2
    
    # RBF kernel: exp(-gamma * dist_sq)
    # Choose a fixed gamma. A value around 1.0 to 2.0 is reasonable for domain [-1,1].
    # Let's use gamma = 1.0 for simplicity.
    gamma = 1.0
    kernel = np.exp(-gamma * dist_sq)  # (N, 10)
    
    # params: (10,)
    # Ensure params is 1D
    params = np.asarray(params, dtype=np.float64)
    
    # Compute prediction: kernel @ params
    pred = kernel @ params
    
    return pred
# EVOLVE-BLOCK-END
