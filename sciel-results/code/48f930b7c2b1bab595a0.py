"""Sum of Gaussians (RBF) model with learned centers and widths for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a sum of 3 Gaussian-like exponential terms.
    
    Model:
    f(x, y, z) = c0 + sum_{i=1}^{3} c_i * exp(-w_i * || [x, y, z] - center_i ||^2)
    
    Parameters (length 10):
    - c0: params[0] (Global offset)
    - For each term i (1, 2, 3):
        - c_i: params[1 + 3*(i-1)] (Amplitude)
        - center_i: (params[2 + 3*(i-1)], params[3 + 3*(i-1)], params[4 + 3*(i-1)])
        - w_i: params[5 + 3*(i-1)] (Width, squared inverse)
    
    This model is non-linear in parameters due to the centers and widths inside the exponent.
    BFGS is used to fit these parameters.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    params = np.asarray(params, dtype=np.float64).ravel()
    if params.size != 10:
        params = np.ones(10)
    
    c0 = params[0]
    
    prediction = np.full_like(x, c0, dtype=np.float64)
    
    # 3 Gaussian terms
    for i in range(3):
        c_i = params[1 + 3*i]
        cx = params[2 + 3*i]
        cy = params[3 + 3*i]
        cz = params[4 + 3*i]
        w_i = params[5 + 3*i]
        
        # Ensure width is positive to avoid explosion
        # Use a softplus-like or absolute value transformation for width
        # w_i is the inverse variance. We want it > 0.
        # Let's use exp(w_i) to ensure positivity, but that might be too large.
        # Or just clip w_i to a small positive number.
        # Given BFGS starts at 1, exp(1) is fine.
        # Let's use np.exp(np.clip(w_i, -10, 10)) to keep it stable.
        width = np.exp(np.clip(w_i, -10.0, 10.0))
        
        # Compute squared distance
        # d2 = (x-cx)^2 + (y-cy)^2 + (z-cz)^2
        dx = x - cx
        dy = y - cy
        dz = z - cz
        d2 = dx*dx + dy*dy + dz*dz
        
        # Gaussian term
        # exp(-width * d2)
        # Clip exponent to avoid underflow/overflow issues in exp, though exp(-large) is 0
        exponent = -width * d2
        exponent = np.clip(exponent, -500.0, 0.0)
        
        term = c_i * np.exp(exponent)
        prediction += term
        
    # Ensure finite output
    if not np.all(np.isfinite(prediction)):
        return np.full_like(x, np.nan)
    
    return prediction
# EVOLVE-BLOCK-END
