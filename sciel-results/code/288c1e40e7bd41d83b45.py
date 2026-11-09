"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Use a rich polynomial basis up to degree 9 (10 coefficients)
    # This allows fitting any polynomial of degree <= 9 exactly if the target is such.
    # The basis functions are x^0, x^1, ..., x^9.
    # We compute the power terms safely.
    
    # x is an array of shape (N,)
    # params is an array of shape (10,)
    
    # Compute powers: 1, x, x^2, ..., x^9
    # We can do this iteratively or using np.vander
    # np.vander(x, N=10, increasing=True) gives columns [1, x, x^2, ..., x^9]
    
    # However, for numerical stability and speed, let's just compute directly.
    # Since x is in [0.1, 6], powers up to 9 are well-behaved (6^9 ~ 10^7),
    # which is within float64 range and precision is generally okay for BFGS.
    
    # Let's use np.vander for clarity and vectorization.
    # vander returns shape (N, 10)
    X = np.vander(x, N=10, increasing=True)
    
    # Prediction is X @ params
    # Shape (N, 10) @ (10,) -> (N,)
    return X @ params
# EVOLVE-BLOCK-END
