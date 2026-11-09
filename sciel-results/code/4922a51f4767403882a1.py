# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Radial Basis Function (RBF) network with fixed centers on [0.1, 6].
    Uses 9 Gaussian kernels with fixed centers and a learnable width parameter.
    y = p0 + sum_{k=1}^{9} p_k * exp(-((x - c_k)^2) / (2 * sigma^2))
    where sigma = softplus(params[9]) to ensure positive width.
    
    This provides a localized, non-polynomial basis that can sharply approximate
    complex hidden functions better than global polynomial expansions.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    N = x.shape[0]
    if N == 0:
        return np.array([], dtype=np.float64)
    
    # Fixed centers spanning the input domain [0.1, 6]
    # Use 9 centers evenly spaced
    centers = np.linspace(0.1, 6.0, 9)
    
    # Learnable width: use softplus to ensure positive sigma
    # sigma = log(1 + exp(params[9]))
    # For numerical stability, clip params[9]
    sigma_param = np.clip(p[9], -100, 100)
    sigma = np.log1p(np.exp(sigma_param))
    
    # Ensure sigma doesn't get too small (avoid division by zero / extreme values)
    sigma = np.maximum(sigma, 1e-6)
    
    # Compute Gaussian RBF values
    # x shape: (N,), centers shape: (9,)
    # diff shape: (N, 9)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # exp(-diff^2 / (2 * sigma^2))
    # Clip the exponent to avoid underflow issues (though exp handles it)
    exponent = -0.5 * (diff / sigma) ** 2
    
    # Clip exponent to [-500, 0] to prevent underflow warnings and improve stability
    # exp(-500) is effectively 0
    exponent = np.clip(exponent, -500, 0)
    
    rbf_values = np.exp(exponent)  # shape: (N, 9)
    
    # Add bias term (p[0])
    y = p[0] + rbf_values @ p[1:10]
    
    return y
# EVOLVE-BLOCK-END
