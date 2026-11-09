"""Symbolic regression with 1 input(s) using a pure Gaussian RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
    
    n = x.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    
    # 10 Gaussian RBFs
    # Centers evenly spaced in [0.1, 6.0]
    centers = np.linspace(0.1, 6.0, 10)
    
    # Bandwidths
    # A bandwidth of 0.7 provides a good balance between local flexibility
    # and global smoothness for the given domain and number of centers.
    sigma = 0.7
    
    # Compute basis matrix (n, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    basis = np.exp(-(diff**2) / (2.0 * sigma**2))
    
    # Linear combination
    pred = basis @ params
    
    # Ensure finite values
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
