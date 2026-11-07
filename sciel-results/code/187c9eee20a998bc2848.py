import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid symbolic regression model combining Gaussian RBFs and a learnable
    polynomial/sinusoidal basis.
    
    params[0]: log(sigma) for RBFs
    params[1:4]: coefficients for polynomial terms (constant, linear, quadratic)
    params[4:6]: amplitude and frequency for a sinusoidal term
    params[6:10]: weights for 4 Gaussian RBFs at fixed centers
    
    This structure allows the model to capture:
    - Smooth global trends via the polynomial part
    - Periodic features via the sinusoidal part
    - Local, non-smooth, or complex features via the RBF part
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # Ensure params is length 10
    if p.shape[0] != 10:
        if p.shape[0] < 10:
            p = np.concatenate([p, np.ones(10 - p.shape[0])])
        else:
            p = p[:10]
    
    # --- RBF Component ---
    log_sigma = np.clip(p[0], -10, 10)
    sigma = np.exp(log_sigma)
    
    # 4 centers evenly spaced in [-3, 3]
    centers = np.linspace(-3.0, 3.0, 4)
    
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    diff_sq = diff * diff
    denom = 2.0 * sigma * sigma
    arg = -diff_sq / denom
    rbf_values = np.exp(arg)
    
    rbf_weights = p[6:10]
    y_rbf = rbf_values @ rbf_weights
    
    # --- Polynomial Component ---
    # Use a simple polynomial basis: 1, x, x^2
    # This provides a smooth global fit
    y_poly = p[1] + p[2] * x + p[3] * x**2
    
    # --- Sinusoidal Component ---
    # One sinusoidal term with learnable amplitude and frequency
    freq = np.clip(p[5], -20, 20)
    amp = p[4]
    y_sin = amp * np.sin(freq * x)
    
    # Combine all components
    result = y_rbf + y_poly + y_sin
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
