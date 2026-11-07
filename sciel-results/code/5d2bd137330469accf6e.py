import numpy as np
import time

# Hard deadline guard: return safely before the 150 s limit.
_START = time.time()
_DEADLINE = _START + 140.0

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluate a linear combination of Gaussian Radial Basis Functions (RBFs).
    
    x: input array (values in [0.1, 6])
    params: array of length 10, weights for 10 Gaussian RBFs
    
    Returns: array of predictions, one per point in x.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    n = x.size
    
    if n == 0:
        return np.empty(0, dtype=np.float64)
    
    p = np.asarray(params, dtype=np.float64)
    if p.ndim != 1 or p.shape[0] != 10:
        return np.zeros_like(x)
    
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.zeros_like(x)
    
    # Define 10 evenly spaced centers across [0.1, 6]
    x_min = 0.1
    x_max = 6.0
    centers = np.linspace(x_min, x_max, 10)
    
    # Fixed width for Gaussian RBFs. 
    # The spacing between centers is (6.0 - 0.1) / 9 = 0.655...
    # A width of ~0.35 provides good overlap for smooth functions while allowing local adaptation.
    # sigma = 0.35
    sigma = 0.35
    sigma_sq = sigma * sigma
    
    # Compute distances squared: (n, 1) - (1, 10) -> (n, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    diff_sq = diff * diff
    
    # Compute Gaussian RBFs: exp(-diff_sq / (2 * sigma_sq))
    # We can precompute the constant factor
    factor = -1.0 / (2.0 * sigma_sq)
    rbfs = np.exp(diff_sq * factor)
    
    # Linear combination
    out = rbfs @ p
    
    if not np.all(np.isfinite(out)):
        return np.zeros_like(x)
    
    return out
