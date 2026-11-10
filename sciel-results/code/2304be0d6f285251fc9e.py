"""Symbolic regression with 1 input: adaptive RBF + polynomial basis."""
import numpy as np
import time

# Global deadline guard
_DEADLINE = time.time() + 140.0


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: linear combination of 10 basis functions of x.
    
    Basis functions:
    - 5 Gaussian RBFs with fixed centers at -3, -1.5, 0, 1.5, 3
      Widths are modulated by params[5:10] (softplus to ensure positivity)
    - 5 polynomial terms (degree 0-4)
    
    The model is linear in the weights (params[0:5]) but non-linear in widths.
    BFGS is used for the full 10-param optimization.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    n = len(x)
    
    # Fixed centers for RBFs
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    # Widths from params[5:10], use softplus to ensure positive widths
    # softplus(z) = log(1 + exp(z))
    # Clip z to avoid overflow in exp
    z = params[5:10]
    z_clipped = np.clip(z, -50.0, 50.0)
    widths = np.log1p(np.exp(z_clipped))
    
    # Ensure minimum width to avoid numerical issues
    widths = np.maximum(widths, 0.1)
    
    # Compute RBF basis: 5 Gaussians
    # RBF_i(x) = exp(-((x - c_i)^2) / (2 * w_i^2))
    rbf = np.zeros((n, 5), dtype=np.float64)
    for i in range(5):
        diff = x - centers[i]
        rbf[:, i] = np.exp(-(diff ** 2) / (2.0 * widths[i] ** 2))
    
    # Polynomial basis: degree 0-4
    poly = np.zeros((n, 5), dtype=np.float64)
    poly[:, 0] = 1.0
    poly[:, 1] = x
    poly[:, 2] = x ** 2
    poly[:, 3] = x ** 3
    poly[:, 4] = x ** 4
    
    # Combine: first 5 params are RBF weights, next 5 are poly coefficients
    # Wait, the strategy says: "five parameters control the weights of these basis functions 
    # and the remaining five parameters modulate the width"
    # So params[0:5] = RBF weights, params[5:10] = width modulators
    
    pred = rbf @ params[0:5] + poly @ params[5:10]
    
    # Safety: replace non-finite values
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
