import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of Gaussian RBFs at fixed centers.
    
    params[0]: log(sigma)
    params[1:10]: weights for 9 RBFs
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Extract log_sigma and weights
    log_sigma = params[0]
    weights = params[1:10]
    
    # Calculate sigma
    # Clamp log_sigma to prevent extreme values that cause underflow/overflow
    # exp(-100) is 0, exp(100) is inf. 
    # If sigma is very small, RBFs become spikes. If very large, they become flat.
    log_sigma = np.clip(log_sigma, -10, 10)
    sigma = np.exp(log_sigma)
    
    # Define 9 centers evenly spaced in [-3, 3]
    centers = np.linspace(-3.0, 3.0, 9)
    
    # Compute RBF values
    # x shape: (N,), centers shape: (9,)
    # diff shape: (N, 9)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Compute squared distances
    diff_sq = diff * diff
    
    # Compute Gaussian kernels
    # exp(-diff_sq / (2 * sigma^2))
    # Handle sigma=0 case (though log_sigma clipping prevents exact 0, but small sigma -> large exponent magnitude)
    # If sigma is very small, 1/sigma^2 is huge.
    # 2 * sigma^2
    denom = 2.0 * sigma * sigma
    
    # To avoid overflow in exp, we can clip the argument
    # arg = -diff_sq / denom
    arg = -diff_sq / denom
    
    # Clip arg to [-700, 700] to prevent exp underflow/overflow warnings/issues
    # exp(-700) is ~1e-304, which is fine. exp(700) is inf. But arg is always <= 0.
    # So we only care about underflow (arg -> -inf).
    # np.exp(-1000) is 0. This is fine.
    
    rbf_values = np.exp(arg)
    
    # Combine with weights
    # weights shape: (9,), rbf_values shape: (N, 9)
    prediction = rbf_values @ weights
    
    return prediction
