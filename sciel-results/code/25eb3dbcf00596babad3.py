"""Symbolic regression: fit a rich linear-in-parameters basis via BFGS."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_BUDGET = 135.0  # safety margin before 150s kill


def _remaining_time() -> float:
    """Return remaining seconds before safety deadline."""
    return _TIME_BUDGET - (time.time() - _START_TIME)


def _softplus_basis(x):
    """
    Compute softplus basis functions with fixed centers and slopes.
    
    Uses 10 softplus units: softplus(s_i * (x - c_i))
    where centers c_i and slopes s_i are fixed on a symmetric grid in [-2, 2].
    
    This creates a piecewise-smooth basis that can approximate a wide range
    of smooth functions while remaining linear in the weights (params).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    
    # Fixed symmetric centers in [-2, 2]
    # 5 positive/negative pairs plus center
    centers = np.array([-2.0, -1.5, -0.75, 0.0, 0.75, 1.5, 2.0, -1.0, 1.0, 0.5])
    
    # Fixed slopes - varying to capture different frequency components
    # Larger slopes = sharper transitions, smaller = smoother
    slopes = np.array([4.0, 3.0, 2.5, 2.0, 1.5, 4.0, 3.0, 3.5, 3.5, 2.0])
    
    # Compute softplus: log(1 + exp(z)) for numerical stability
    basis = np.zeros((n, 10))
    for i in range(10):
        z = slopes[i] * (x - centers[i])
        # Numerically stable softplus
        # For z > 30: softplus(z) ≈ z
        # For z < -30: softplus(z) ≈ exp(z)
        # Otherwise: log(1 + exp(z))
        softplus_vals = np.zeros_like(z)
        
        # Large positive: linear
        mask_pos = z > 30
        softplus_vals[mask_pos] = z[mask_pos]
        
        # Large negative: exponential
        mask_neg = z < -30
        softplus_vals[mask_neg] = np.exp(z[mask_neg])
        
        # Middle range: use log1p(exp())
        mask_mid = ~(mask_pos | mask_neg)
        if np.any(mask_mid):
            z_mid = z[mask_mid]
            softplus_vals[mask_mid] = np.log1p(np.exp(z_mid))
        
        basis[:, i] = softplus_vals
    
    return basis


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluate the model: sum of params[i] * softplus(s_i * (x - c_i)).
    This is linear in params, so BFGS should converge to the global minimum.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    # Ensure params has exactly 10 elements
    if len(params) < 10:
        # Pad with zeros if needed
        params = np.concatenate([params, np.zeros(10 - len(params))])
    else:
        params = params[:10]
    
    basis = _softplus_basis(x)
    return basis @ params
# EVOLVE-BLOCK-END
