"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = None
_TIME_LIMIT = 115.0  # Safety margin before 120s kill

def _check_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_LIMIT
    return time.time() < _DEADLINE


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: combines a rich fixed basis (polynomials, exp, log, trig, rational)
    with an adaptive RBF component. The RBF component uses a learnable width (sigma)
    and 4 learnable weights, while the fixed basis uses 5 weights. Total: 10 params.
    
    This structure allows the model to capture both global trends (via the fixed basis)
    and local variations (via the RBF), which is more flexible than either alone.
    
    Args:
        x: Input array of shape (N,).
        params: Array of 10 parameters.
        
    Returns:
        Prediction array of shape (N,).
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Check deadline
    if not _check_deadline():
        # Return a simple fallback if we're running out of time
        return np.ones_like(x) * params[0]
    
    # Ensure x is positive for log (domain is [0.1, 6] so this is safe,
    # but we clip for numerical safety during optimization)
    x_safe = np.clip(x, 1e-6, None)
    
    # --- Fixed Basis (5 parameters) ---
    # Basis functions:
    # 0: 1 (constant)
    # 1: x (linear)
    # 2: x^2 (quadratic)
    # 3: exp(-x) (decaying exponential)
    # 4: log(1 + x) (logarithmic)
    
    # Extract fixed basis weights
    w_const = params[0]
    w_lin = params[1]
    w_quad = params[2]
    w_exp = params[3]
    w_log = params[4]
    
    # Compute fixed basis contribution
    y_fixed = (w_const * 1.0 + 
               w_lin * x + 
               w_quad * x**2 + 
               w_exp * np.exp(np.clip(-x, -50, 50)) + 
               w_log * np.log1p(x_safe))
    
    # --- RBF Component (5 parameters) ---
    # 1 learnable width (sigma) + 4 RBF weights
    # Fixed centers: 4 centers evenly spaced in [0.1, 6]
    
    n_rbf = 4
    centers = np.linspace(0.1, 6.0, n_rbf)
    
    # Extract width parameter (sigma) from params[5]
    # Use softplus-like transformation: sigma = exp(params[5])
    # Clip to reasonable range to avoid numerical issues
    sigma = np.exp(params[5])
    sigma = np.clip(sigma, 0.05, 5.0)
    
    # Extract RBF weights
    rbf_weights = params[6:10]
    
    # Compute RBF basis: phi(x) = exp(-((x - c)^2) / (2 * sigma^2))
    # x shape: (N,), centers shape: (n_rbf,)
    # We want basis shape: (N, n_rbf)
    
    diff = x[:, np.newaxis] - centers[np.newaxis, :]  # (N, n_rbf)
    sq_dist = diff * diff  # (N, n_rbf)
    
    # Compute Gaussian RBF values
    rbf_basis = np.exp(-sq_dist / (2.0 * sigma * sigma))
    
    # Check for non-finite values
    if not np.all(np.isfinite(rbf_basis)):
        rbf_basis = np.nan_to_num(rbf_basis, nan=0.0, posinf=0.0, neginf=0.0)
    
    # RBF contribution
    y_rbf = rbf_basis @ rbf_weights
    
    # Total prediction
    result = y_fixed + y_rbf
    
    # Ensure output is finite
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return result
# EVOLVE-BLOCK-END
