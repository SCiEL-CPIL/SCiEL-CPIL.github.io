# EVOLVE-BLOCK-START
import numpy as np
import time

def kakeya_needle_3d8() -> np.ndarray:
    """
    Minimizes the union volume of 64 sheared tubes in 3D.
    Uses a 'Centered V-Shape' packing strategy where base positions are chosen
    to minimize the width of the union at c=0.5, creating a V-shaped profile
    for the union width as a function of height c.
    """
    N = 8
    # Base positions x_k, y_k
    # x_k = -i/16 - 1/16
    # y_k = -j/16 - 1/16
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = -i_flat / 16.0 - 1.0 / 16.0
    base_y = -j_flat / 16.0 - 1.0 / 16.0
    
    positions = np.stack([base_x, base_y], axis=1)
    return positions

# EVOLVE-BLOCK-END
