"""Combined program: Hybrid RBF basis with fixed centers and learned widths."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid RBF model with 5 fixed centers and 5 learnable widths.
    
    Structure:
    - 5 centers evenly spaced in [-3, 3]: -3, -1.5, 0, 1.5, 3
    - 5 parameters (params[0:5]) are weights for each RBF
    - 5 parameters (params[5:10]) are log-widths (log(sigma)) for each RBF
    
    This makes the model non-linear in params (widths are in the exponent),
    but BFGS can handle this. The structure allows localized adaptation
    while maintaining smoothness.
    
    We also add a small polynomial baseline (constant + linear) using two
    of the weight parameters? No, we have exactly 10 params. Let's use:
    - params[0:5]: weights for the 5 RBFs
    - params[5:10]: log(sigma) for each of the 5 RBFs
    
    This uses all 10 parameters meaningfully.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure params is 1D and length 10
    if params.ndim != 1:
        params = params.ravel()
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)), constant_values=0.0)
    else:
        params = params[:10]
    
    # Fixed centers: 5 evenly spaced points in [-3, 3]
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    # Extract weights and log-widths
    weights = params[0:5]
    log_sigmas = params[5:10]
    
    # Convert log-widths to sigmas, clamp to avoid numerical issues
    # sigma = exp(log_sigma), but clamp log_sigma to [-5, 5] to keep sigma in [~0.0067, ~148]
    log_sigmas_clamped = np.clip(log_sigmas, -5.0, 5.0)
    sigmas = np.exp(log_sigmas_clamped)
    
    # Compute RBF values: exp(-0.5 * (x - c)^2 / sigma^2)
    # x shape: (N,), centers shape: (5,)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]  # (N, 5)
    
    # sigma shape: (5,) -> (1, 5) for broadcasting
    sigma2 = sigmas[np.newaxis, :] ** 2
    
    # Avoid division by zero (though sigma >= exp(-5) ~ 0.0067, so safe)
    rbf_values = np.exp(-0.5 * diff ** 2 / sigma2)  # (N, 5)
    
    # Weighted sum
    result = rbf_values @ weights  # (N,)
    
    # Safety: replace non-finite values
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
