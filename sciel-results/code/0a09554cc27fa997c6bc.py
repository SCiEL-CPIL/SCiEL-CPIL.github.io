import numpy as np

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    # Optimal base positions derived from minimizing the union volume
    # X_i = -i / (2*N) and Y_j = -j / (2*N)
    # This centers the shear so that all x-intervals overlap at c=0.5
    i_vals = np.arange(N)
    j_vals = np.arange(N)
    
    # Create meshgrid for i and j
    i_grid, j_grid = np.meshgrid(i_vals, j_vals, indexing="ij")
    
    # Calculate optimal X and Y
    X = -i_grid / (2.0 * N)
    Y = -j_grid / (2.0 * N)
    
    # Flatten to (64, 2) array
    # Row k corresponds to i = k // 8, j = k % 8
    # X.flatten() and Y.flatten() follow row-major order (C-order)
    # which maps (i, j) to i*N + j. This matches k = i*8 + j.
    positions = np.stack([X.flatten(), Y.flatten()], axis=1)
    
    return positions
