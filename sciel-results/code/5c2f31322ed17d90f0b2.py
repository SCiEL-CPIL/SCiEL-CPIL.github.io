"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian RBF network with 10 parameters as centers.
    Fixed multi-scale widths and amplitudes.
    
    Design:
    - 10 centers optimized by BFGS.
    - Fixed widths: Multi-scale set [0.5, 1.0, 1.5, 2.0, 2.5] repeated twice.
      Wider widths than the previous 0.1-2.0 to ensure smoother gradients for BFGS.
    - Fixed amplitudes: Alternating signs [1, -1, 1, -1, ...] to allow the model
      to fit functions that take both positive and negative values, and to create
      valleys (subtractive bumps) in addition to peaks.
    """
    centers = params
    
    # Fixed widths: Multi-scale, avoiding very narrow widths that cause ill-conditioning
    base_widths = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    widths = np.tile(base_widths, 2)
    
    # Fixed amplitudes: Alternating signs to allow positive and negative contributions
    amplitudes = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    
    # Compute RBF outputs
    # x shape: (N,)
    # centers shape: (10,)
    # We want (N, 10) matrix of differences
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Scale by widths
    # widths shape: (10,)
    scaled_diff = diff / widths[np.newaxis, :]
    
    # Compute exponent
    # Clip to prevent underflow issues (though exp handles 0 fine, large negatives are 0)
    # exp(-0.5 * scaled_diff^2)
    rbf_values = np.exp(-0.5 * scaled_diff ** 2)
    
    # Weighted sum
    # amplitudes shape: (10,)
    predictions = np.sum(rbf_values * amplitudes[np.newaxis, :], axis=1)
    
    return predictions
# EVOLVE-BLOCK-END
