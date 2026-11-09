"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series approximation with fixed orthogonal frequencies.
    
    Domain: [-2, 2], length 4. Fundamental frequency for orthogonality: pi/2.
    
    With 10 params, we use:
      params[0]: constant
      params[1]: sin(pi/2 * x)
      params[2]: cos(pi/2 * x)
      params[3]: sin(pi * x)
      params[4]: cos(pi * x)
      params[5]: sin(3*pi/2 * x)
      params[6]: cos(3*pi/2 * x)
      params[7]: sin(2*pi * x)
      params[8]: cos(2*pi * x)
      params[9]: sin(5*pi/2 * x)
    
    This gives complete orthogonal pairs for k=1..4, plus the 5th harmonic sine.
    All frequencies are orthogonal on [-2, 2].
    """
    # x is 1D array of shape (n,)
    x = np.asarray(x, dtype=float)
    
    # Precompute all basis functions at once for efficiency
    # Frequencies: k * pi/2 for k = 1, 2, 3, 4, 5
    k_vals = np.array([1, 2, 3, 4, 5], dtype=float)
    freqs = k_vals * np.pi / 2  # shape (5,)
    
    # Compute sin and cos for all frequencies at once
    # freqs[:, None] * x[None, :] gives shape (5, n)
    angles = np.outer(freqs, x)  # shape (5, n)
    sin_vals = np.sin(angles)    # shape (5, n)
    cos_vals = np.cos(angles)    # shape (5, n)
    
    # Build output: constant + sum of weighted basis functions
    out = np.full_like(x, params[0], dtype=float)
    
    # k=1: sin -> params[1], cos -> params[2]
    out += params[1] * sin_vals[0]
    out += params[2] * cos_vals[0]
    
    # k=2: sin -> params[3], cos -> params[4]
    out += params[3] * sin_vals[1]
    out += params[4] * cos_vals[1]
    
    # k=3: sin -> params[5], cos -> params[6]
    out += params[5] * sin_vals[2]
    out += params[6] * cos_vals[2]
    
    # k=4: sin -> params[7], cos -> params[8]
    out += params[7] * sin_vals[3]
    out += params[8] * cos_vals[3]
    
    # k=5: sin -> params[9]
    out += params[9] * sin_vals[4]
    
    return out
# EVOLVE-BLOCK-END
