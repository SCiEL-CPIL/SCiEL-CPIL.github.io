"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 1 input.
    Uses a Radial Basis Function (RBF) model with 3 Gaussian kernels.
    
    Parameters (length 10):
    0: offset (bias)
    1-3: centers of the 3 Gaussians
    4-6: log-widths of the 3 Gaussians (sigma = exp(param))
    7-9: weights of the 3 Gaussians
    
    Model:
    f(x) = offset + sum_{i=1}^{3} weight_i * exp(-0.5 * ((x - center_i) / sigma_i)^2)
    """
    x = np.asarray(x, dtype=float)
    if x.ndim > 1:
        x = x.ravel()
    
    params = np.asarray(params, dtype=float)
    if params.shape[0] != 10:
        # Fallback to safe values if shape is wrong, though evaluator ensures 10
        params = np.ones(10)
    
    offset = params[0]
    
    centers = params[1:4]
    log_sigmas = params[4:7]
    weights = params[7:10]
    
    # Ensure sigmas are positive and not too small to avoid division by zero/overflow
    # exp(log_sigma) ensures positivity.
    # We clip log_sigmas to prevent extreme values causing overflow in exp or underflow in the Gaussian.
    # If log_sigma is very large, sigma is huge, term is ~1.
    # If log_sigma is very small (negative), sigma is tiny, term is spike.
    # Clipping log_sigmas to [-10, 10] keeps sigmas in [4.5e-5, 22026], which is safe.
    log_sigmas = np.clip(log_sigmas, -10, 10)
    sigmas = np.exp(log_sigmas)
    
    prediction = np.full_like(x, offset, dtype=float)
    
    for i in range(3):
        c = centers[i]
        s = sigmas[i]
        w = weights[i]
        
        # Calculate Gaussian term
        # (x - c) / s
        z = (x - c) / s
        # exp(-0.5 * z^2)
        # Clip z^2 to avoid overflow in exp if z is huge (though z is bounded by x range / min sigma)
        # x in [-3, 3], c in R. If c is far away, z is large.
        # exp(-0.5 * z^2) underflows to 0 for z > ~38. This is fine.
        term = np.exp(-0.5 * z * z)
        prediction += w * term
        
    # Ensure finite values
    prediction = np.where(np.isfinite(prediction), prediction, 0.0)
    
    return prediction
# EVOLVE-BLOCK-END
