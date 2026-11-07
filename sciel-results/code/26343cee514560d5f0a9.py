"""RBF-based symbolic regression with 10 fixed centers and shared width."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10 RBFs at fixed centers in [-3, 3], shared width param[0],
    weights params[1:10], offset params[10]... wait, params has length 10.
    
    Actually params has length 10. Let's use:
    - params[0]: width (sigma) for all RBFs
    - params[1:9]: 8 RBF weights
    - params[9]: offset
    
    That's 1 + 8 + 1 = 10 parameters.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # 10 evenly spaced centers in [-3, 3]
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Shared width: use exp(params[0]) to keep it positive, starting near 1
    sigma = np.exp(params[0])
    
    # Weights for 8 RBFs (indices 1-8), offset at index 9
    weights = params[1:9]
    offset = params[9]
    
    # Compute RBF values: exp(-((x - c)^2) / (2 * sigma^2))
    # x shape: (N,), centers shape: (10,)
    # We want (N, 10) matrix of distances
    diffs = x[:, np.newaxis] - centers[np.newaxis, :]  # (N, 10)
    
    # Use only first 8 centers to match 8 weights
    diffs_8 = diffs[:, :8]  # (N, 8)
    
    rbf_vals = np.exp(-(diffs_8 ** 2) / (2.0 * sigma ** 2))  # (N, 8)
    
    prediction = rbf_vals @ weights + offset
    
    return prediction
# EVOLVE-BLOCK-END
