"""Structural change: Use a 9th-degree polynomial basis on scaled input to capture non-linear smooth functions."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x to [-1, 1] for better numerical conditioning of the polynomial basis
    # Original range is [-2, 2]
    xs = x / 2.0
    
    # Build the Vandermonde matrix for powers 0 to 9
    # params[0] * 1 + params[1] * xs + ... + params[9] * xs^9
    # Using np.vander is efficient
    # vander(xs, N=10, increasing=True) gives columns [1, xs, xs^2, ..., xs^9]
    basis = np.vander(xs, N=10, increasing=True)
    
    # Linear combination
    return basis @ params
# EVOLVE-BLOCK-END
