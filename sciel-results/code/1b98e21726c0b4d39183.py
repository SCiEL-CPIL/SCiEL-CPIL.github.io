"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Radial Basis Function (RBF) network with 5 centers and 5 widths.
    params[0:5] are the centers c_i
    params[5:10] are the widths sigma_i (absolute value used to ensure positivity)
    
    The output is the sum of 5 Gaussian kernels:
    y = sum_i exp(-(x - c_i)^2 / sigma_i^2)
    
    Weights are fixed to 1.
    To break symmetry during optimization (starting from all ones),
    we add small distinct offsets to the centers and widths.
    """
    centers_raw = params[0:5]
    widths_raw = params[5:10]
    
    # Symmetry breaking offsets
    # These are small enough to not significantly bias the final solution
    # but large enough to distinguish the kernels at initialization.
    center_offsets = np.array([0.0, 0.1, -0.1, 0.2, -0.2])
    width_offsets = np.array([0.0, 0.05, -0.05, 0.1, -0.1])
    
    centers = centers_raw + center_offsets
    
    # Ensure widths are positive and stable
    # Using abs ensures positivity. Adding a small epsilon prevents division by zero.
    # We also add the offset to width_raw before abs? 
    # If width_raw is 1.0, width = abs(1.0 + 0.0) = 1.0
    # If width_raw is 1.0, width = abs(1.0 + 0.05) = 1.05
    # This helps break symmetry in widths too.
    widths = np.abs(widths_raw + width_offsets) + 1e-6
    
    # Compute RBF terms
    # x shape: (N,)
    # centers shape: (5,)
    # We need (N, 5) differences
    x_reshaped = x[:, np.newaxis]
    centers_reshaped = centers[np.newaxis, :]
    
    diffs = x_reshaped - centers_reshaped
    
    # sigmas shape: (5,)
    sigmas_reshaped = widths[np.newaxis, :]
    
    # Gaussian kernel
    # exp(-diffs^2 / sigmas^2)
    kernel_values = np.exp(-(diffs ** 2) / (sigmas_reshaped ** 2))
    
    # Sum of kernels (weights are 1)
    y = np.sum(kernel_values, axis=1)
    
    return y
# EVOLVE-BLOCK-END
