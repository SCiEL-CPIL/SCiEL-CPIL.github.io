import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    # Extract parameters
    amps = params[0:5]
    widths_raw = params[5:10]
    
    # Ensure positive widths
    widths = np.abs(widths_raw) + 0.1
    
    # Fixed centers
    centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # Compute Gaussians
    # x is (n,), centers is (5,)
    # We want shape (n, 5)
    x_expanded = x[:, np.newaxis]
    c_expanded = centers[np.newaxis, :]
    w_expanded = widths[np.newaxis, :]
    
    # Exponent term: -0.5 * ((x - c) / w)^2
    diff = (x_expanded - c_expanded)
    # Avoid division by zero if w is very small, but w >= 0.1
    scaled_diff = diff / w_expanded
    exponent = -0.5 * scaled_diff**2
    
    # Gaussian values
    gaussians = np.exp(exponent)
    
    # Weighted sum
    # amps is (5,), gaussians is (n, 5)
    result = gaussians @ amps
    
    return result
