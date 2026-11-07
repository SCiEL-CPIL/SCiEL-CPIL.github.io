"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Adaptive RBF + polynomial hybrid model.
    
    Structural change: Instead of a fixed Chebyshev basis, we use a model
    where 5 RBF centers are learnable parameters. This allows the model
    to adapt its shape to the target function's structure.
    
    Params layout (length 10):
      - params[0:5]  : RBF centers (squashed to [-2, 2] via tanh)
      - params[5:10] : RBF amplitudes
    
    The model is:
      y(x) = sum_i  amp_i * exp(-((x - center_i)^2) / (2 * sigma^2))
    
    where sigma is a fixed bandwidth (0.5) chosen for good resolution
    on [-2, 2].
    
    BFGS will jointly optimize centers and amplitudes.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Squash centers to [-2, 2] for stability
    centers = 2.0 * np.tanh(params[0:5])
    amplitudes = params[5:10]
    
    # Fixed bandwidth for good resolution on [-2, 2]
    sigma = 0.5
    
    result = np.zeros_like(x)
    for i in range(5):
        diff = x - centers[i]
        result += amplitudes[i] * np.exp(-0.5 * (diff / sigma) ** 2)
    
    return result
# EVOLVE-BLOCK-END
