"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a piecewise-linear basis (hat functions).
    This local basis is well-conditioned for BFGS optimization and can represent
    arbitrary continuous functions defined on the domain [-3, 3].
    
    We use 10 parameters corresponding to nodes at x = -3, -2.33, ..., 3.
    The prediction is the linear combination of these hat functions.
    """
    # Ensure x is a numpy array
    x = np.asarray(x, dtype=np.float64)
    
    # Number of basis functions (parameters)
    n_basis = 10
    
    # Fixed centers for the hat functions, uniformly spaced in [-3, 3]
    # Using linspace ensures coverage of the entire domain
    centers = np.linspace(-3.0, 3.0, n_basis)
    
    # Compute the support width for each hat function
    # For interior points, the width is the distance to the nearest neighbor / 2 * 2 = distance to neighbor
    # Actually, standard hat functions have support [c_{i-1}, c_{i+1}]
    # Let's define the basis functions explicitly.
    
    # To be efficient, we can compute the basis matrix B of shape (N, n_basis)
    # B[i, j] = hat_j(x[i])
    
    # Initialize basis matrix
    N = x.shape[0]
    basis = np.zeros((N, n_basis), dtype=np.float64)
    
    # Calculate distances between centers to determine slopes/widths
    # Since centers are uniform, the width is constant.
    if n_basis > 1:
        width = centers[1] - centers[0]
    else:
        width = 1.0
        
    # For each point x, find which interval it falls into or compute directly.
    # Direct computation is safer and fast enough for N=200, K=10.
    
    for j in range(n_basis):
        c = centers[j]
        
        # Determine the left and right boundaries for this hat function
        if j == 0:
            left = c
            right = c + width
            # Left side: linear ramp up from 0 to 1 over [left, c]
            # Right side: linear ramp down from 1 to 0 over [c, right]
            # For x < left: 0
            # For left <= x <= c: (x - left) / (c - left)
            # For c < x <= right: (right - x) / (right - c)
            # For x > right: 0
            
            # Vectorized calculation for column j
            # Left part
            mask_left = (x >= left) & (x <= c)
            basis[mask_left, j] = (x[mask_left] - left) / (c - left)
            
            # Right part
            mask_right = (x > c) & (x <= right)
            basis[mask_right, j] = (right - x[mask_right]) / (right - c)
            
        elif j == n_basis - 1:
            left = c - width
            right = c
            # Left side: linear ramp up from 0 to 1 over [left, c]
            # Right side: linear ramp down from 1 to 0 over [c, right]
            # For x < left: 0
            # For left <= x <= c: (x - left) / (c - left)
            # For c < x <= right: (right - x) / (right - c)
            # For x > right: 0
            
            # Vectorized calculation for column j
            # Left part
            mask_left = (x >= left) & (x <= c)
            basis[mask_left, j] = (x[mask_left] - left) / (c - left)
            
            # Right part
            mask_right = (x > c) & (x <= right)
            basis[mask_right, j] = (right - x[mask_right]) / (right - c)
            
        else:
            left = centers[j-1]
            right = centers[j+1]
            # For x < left: 0
            # For left <= x <= c: (x - left) / (c - left)
            # For c < x <= right: (right - x) / (right - c)
            # For x > right: 0
            
            # Vectorized calculation for column j
            # Left part
            mask_left = (x >= left) & (x <= c)
            basis[mask_left, j] = (x[mask_left] - left) / (c - left)
            
            # Right part
            mask_right = (x > c) & (x <= right)
            basis[mask_right, j] = (right - x[mask_right]) / (right - c)

    # Linear combination of basis functions
    # y = basis @ params
    y = basis.dot(params)
    
    return y
# EVOLVE-BLOCK-END
