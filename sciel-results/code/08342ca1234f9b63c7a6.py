# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of 10 localized Radial Basis Functions (Gaussian).
    
    Each basis function is centered at a fixed location in [-3, 3] with a fixed width.
    The 10 parameters scale each basis function independently.
    
    This design allows the model to capture local non-linear features that a global
    linear model cannot, while keeping the optimization problem linear in parameters,
    which is well-suited for BFGS.
    """
    # Fixed centers spread across the input domain [-3, 3]
    centers = np.array([
        -3.0, -2.25, -1.5, -0.75, 0.0,
        0.75, 1.5, 2.25, 3.0, 0.0
    ])
    
    # Fixed widths (sigma) for the Gaussian RBFs.
    # We use a mix of narrow and wide widths to capture both local and global features.
    # Note: The last center is 0.0 again but with a different width to add flexibility at the center.
    sigmas = np.array([
        0.8, 0.8, 0.8, 0.8, 0.8,
        0.8, 0.8, 0.8, 0.8, 1.5
    ])
    
    # Compute the basis functions: phi_i(x) = exp(-((x - c_i)^2) / (2 * sigma_i^2))
    # x shape: (N, 1) or (N,)
    # We want output shape (N,)
    
    # Reshape for broadcasting: x is (N, 1), centers is (1, 10), sigmas is (1, 10)
    x_col = x.reshape(-1, 1)
    c_row = centers.reshape(1, -1)
    s_row = sigmas.reshape(1, -1)
    
    # Calculate squared distances
    diff = x_col - c_row  # (N, 10)
    sq_dist = diff ** 2    # (N, 10)
    
    # Calculate Gaussian values
    # (N, 10) / (1, 10) -> (N, 10)
    gaussians = np.exp(-sq_dist / (2.0 * s_row ** 2))
    
    # Linear combination with params
    # params shape: (10,)
    # result shape: (N, 1) -> (N,)
    result = gaussians @ params
    
    return result
# EVOLVE-BLOCK-END
