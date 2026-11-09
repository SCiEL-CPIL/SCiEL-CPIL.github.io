import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale input to [-1, 1] for numerical stability
    t = x / 3.0
    
    # Linear trend
    # params[0] is intercept, params[1] is slope
    y = params[0] + params[1] * t
    
    # RBF part
    # Fixed centers in scaled domain
    # 8 centers evenly spaced
    centers = np.array([-0.875, -0.625, -0.375, -0.125, 0.125, 0.375, 0.625, 0.875])
    sigma = 0.30
    
    # Compute RBF values
    # t shape (N,), centers shape (8,)
    # We want (N, 8)
    diff = t[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian kernel: exp(-0.5 * (diff/sigma)^2)
    # Pre-compute constant 0.5 / sigma^2
    const = 0.5 / (sigma * sigma)
    rbf_vals = np.exp(-const * diff**2)
    
    # params[2:10] are the amplitudes
    # rbf_vals @ params[2:10] gives (N,)
    y += rbf_vals @ params[2:10]
    
    return y
