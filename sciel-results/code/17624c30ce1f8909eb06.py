"""Symbolic regression with Fourier series basis for global optimization compatibility."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Fourier series basis.
    
    Basis functions (10 total):
    1:  1 (DC offset)
    2-6: sin(k*pi*x/3) for k=1..5 (odd harmonics)
    7-10: cos(k*pi*x/3) for k=1..4 (even harmonics, excluding k=5 to keep 10 terms)
    
    This is a linear-in-parameters model, so BFGS will converge to the global
    minimum for the given frequency set. The frequencies are chosen to be
    orthogonal on [-3, 3] with period 6 (so k*pi*x/3 completes k half-cycles
    over the interval), providing good coverage of both low and medium
    frequency components.
    
    The Fourier basis is well-conditioned and can represent any reasonably
    smooth periodic or quasi-periodic function on the interval.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    n = x.shape[0]
    
    # DC offset
    b1 = np.ones(n)
    
    # Sine terms: sin(k*pi*x/3) for k=1..5
    # These are orthogonal on [-3, 3]
    b2 = np.sin(np.pi * x / 3.0)
    b3 = np.sin(2.0 * np.pi * x / 3.0)
    b4 = np.sin(3.0 * np.pi * x / 3.0)
    b5 = np.sin(4.0 * np.pi * x / 3.0)
    b6 = np.sin(5.0 * np.pi * x / 3.0)
    
    # Cosine terms: cos(k*pi*x/3) for k=1..4
    b7 = np.cos(np.pi * x / 3.0)
    b8 = np.cos(2.0 * np.pi * x / 3.0)
    b9 = np.cos(3.0 * np.pi * x / 3.0)
    b10 = np.cos(4.0 * np.pi * x / 3.0)
    
    # Combine all basis functions with parameters
    result = (params[0] * b1 + 
              params[1] * b2 + 
              params[2] * b3 + 
              params[3] * b4 + 
              params[4] * b5 + 
              params[5] * b6 + 
              params[6] * b7 + 
              params[7] * b8 + 
              params[8] * b9 + 
              params[9] * b10)
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.zeros_like(x)
    
    return result
# EVOLVE-BLOCK-END
