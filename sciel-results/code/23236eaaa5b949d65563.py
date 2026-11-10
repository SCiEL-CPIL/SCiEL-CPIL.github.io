import numpy as np


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    We use an analytical solution that minimizes the volume of the union of the tubes.
    The problem decomposes into two independent 1D problems for x and y coordinates.
    For the 1D problem, we have N intervals of length 1/N sheared by c*i/N.
    The optimal base positions X_i are chosen to minimize the average width of the union.
    This is achieved by setting X_i = -i / (2N), which centers the shear effect.
    
    Returns:
        positions: np.ndarray of shape (64, 2).
    """
    N = 8
    # Optimal 1D base positions
    # X_i = -i / (2 * N)
    base_x = -np.arange(N) / (2 * N)
    base_y = -np.arange(N) / (2 * N)
    
    # Create meshgrid for i and j
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    
    # Flatten to get 64 positions
    base_x_flat = base_x[i_grid.flatten()]
    base_y_flat = base_y[j_grid.flatten()]
    
    positions = np.stack([base_x_flat, base_y_flat], axis=1)
    return positions
