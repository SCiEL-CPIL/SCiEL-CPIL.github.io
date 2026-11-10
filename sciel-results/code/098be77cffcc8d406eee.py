"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Fourier series basis on [-3, 3]
    # x is assumed to be a 1D array of shape (n,)
    # params has length 10: 5 cosine weights + 5 sine weights (or interleaved)
    # Let's use params[0:5] for cos coefficients and params[5:10] for sin coefficients
    # Basis: cos(k * pi * x / 3) and sin(k * pi * x / 3) for k=1..5
    
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure params has at least 10 elements
    if len(params) < 10:
        # Pad if necessary, though the evaluator provides exactly 10
        padded = np.ones(10)
        padded[:len(params)] = params
        params = padded
    
    cos_coefs = params[0:5]
    sin_coefs = params[5:10]
    
    # Precompute the argument
    arg = np.pi * x / 3.0
    
    # Compute the series
    result = np.zeros_like(x)
    
    for k in range(5):
        result += cos_coefs[k] * np.cos((k + 1) * arg)
        result += sin_coefs[k] * np.sin((k + 1) * arg)
    
    return result
# EVOLVE-BLOCK-END
