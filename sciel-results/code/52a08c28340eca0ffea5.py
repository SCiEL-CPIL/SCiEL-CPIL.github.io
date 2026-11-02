"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 600

# EVOLVE-BLOCK-START
def construct():
    # Try a smoothed/triangular profile to see if it reduces the peak
    # of the autoconvolution relative to normalization
    # Use a cosine-bell shape which is smooth and nonnegative
    x = np.linspace(0, 1, N)
    # Cosine bell: zero at endpoints, max at center
    heights = 0.5 * (1 + np.cos(2 * np.pi * x))
    # Ensure nonnegative and avoid exact zeros at endpoints for numerical safety
    heights = np.maximum(heights, 1e-10)
    return heights.tolist()
# EVOLVE-BLOCK-END
