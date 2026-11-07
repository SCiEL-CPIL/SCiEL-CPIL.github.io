"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 50

# EVOLVE-BLOCK-START
def construct():
    # Use a raised cosine (Hann window) shape to reduce autocorrelation peak
    # This spreads energy more evenly in the autocorrelation
    t = np.linspace(0, 1, N)
    # Hann window: 0.5 * (1 - cos(2*pi*t))
    h = 0.5 * (1 - np.cos(2 * np.pi * t))
    # Ensure nonnegative and finite
    h = np.maximum(h, 0.0)
    # Scale to have reasonable magnitude
    h = h * 2.0  # Scale up since Hann has mean 0.5
    return list(h)
# EVOLVE-BLOCK-END
