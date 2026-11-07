"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Radial Basis Function (RBF) network with 4 centers and 4 widths, 
    plus a global scale and a bias.
    
    params[0:4] are the centers c_i
    params[4:8] are the widths sigma_i (absolute value used to ensure positivity)
    params[8] is the global scale W
    params[9] is the bias B
    
    The output is:
    y = W * sum_i exp(-(x - c_i)^2 / sigma_i^2) + B
    
    This allows the model to:
    1. Place 4 local Gaussian bumps at adjustable locations.
    2. Adjust the width of each bump.
    3. Scale the entire sum (allowing negative amplitudes if W < 0).
    4. Shift the entire function vertically.
    
    Starting from all ones:
    Centers at 1.0, Widths at 1.0, Scale 1.0, Bias 1.0.
    Symmetry in centers is an issue, but BFGS should separate them if the target
    has distinct features or if the gradient pushes them apart.
    To help, we add fixed offsets to the centers to break symmetry 
    during optimization, ensuring distinct gradients for each center term.
    """
    # Extract params
    centers = params[0:4]
    widths = np.abs(params[4:8]) + 1e-6 # Ensure positive widths
    scale = params[8]
    bias = params[9]
    
    # Break symmetry by adding fixed offsets to the centers
    # This helps BFGS escape the symmetric local minimum where all centers are identical
    offsets = np.array([-1.5, -0.5, 0.5, 1.5])
    centers = centers + offsets
    
    # Compute RBF terms
    # x shape: (N,)
    # centers shape: (4,)
    x_reshaped = x[:, np.newaxis]
    centers_reshaped = centers[np.newaxis, :]
    widths_reshaped = widths[np.newaxis, :]
    
    diffs = x_reshaped - centers_reshaped
    
    # Gaussian kernel
    kernel_values = np.exp(-(diffs ** 2) / (widths_reshaped ** 2))
    
    # Sum of kernels
    rbf_sum = np.sum(kernel_values, axis=1)
    
    # Apply scale and bias
    y = scale * rbf_sum + bias
    
    return y
# EVOLVE-BLOCK-END
