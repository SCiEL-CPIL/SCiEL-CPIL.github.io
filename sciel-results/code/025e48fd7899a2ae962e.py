import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x).flatten()
    # Normalize x to [-1, 1] to improve numerical stability of high-degree polynomials
    x_norm = x / 2.0
    # Compute powers: 1, x, x^2, ..., x^9
    powers = np.vander(x_norm, N=10, increasing=True)
    return powers @ params
