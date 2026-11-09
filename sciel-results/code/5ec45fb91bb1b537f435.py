"""
Strategy: Gaussian Mixture Model (GMM) style non-linear parameterization.
Instead of a linear basis expansion (e.g., polynomials, sines), we model the
function as a sum of 3 Gaussian bumps plus a bias.
The 10 parameters are:
  - bias (1 param)
  - 3 Gaussians, each with (center, width, amplitude) (9 params)
  
Total: 10 params.
This allows BFGS to fit localized features (peaks, valleys, steps) more flexibly
than a global polynomial, especially if the hidden function has non-smooth or
localized structure.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the hidden function as a sum of 3 Gaussian bumps + bias.
    params[0] = bias
    params[1:4] = center, width, amplitude for Gaussian 1
    params[4:7] = center, width, amplitude for Gaussian 2
    params[7:10] = center, width, amplitude for Gaussian 3
    """
    x = np.asarray(x, dtype=float)
    
    # Ensure params are finite and valid
    if not np.all(np.isfinite(params)):
        return np.zeros_like(x)
        
    # Safety: prevent division by zero or extreme values in Gaussian exponent
    # Widths must be positive. We use softplus or abs+eps to ensure stability.
    # However, BFGS can move widths to negative values. 
    # To keep the function differentiable and stable, we use width = exp(p) or abs(p)+eps.
    # Using exp(p) ensures width > 0 and is smooth. But exp(p) can overflow if p is large.
    # Let's use a safer transformation: width = 1.0 / (1.0 + np.exp(-np.abs(p)))? No, that's sigmoid-like.
    # Better: width = np.abs(p) + 1e-6. This is non-differentiable at 0.
    # Best for BFGS: width = np.exp(p). If p is large positive, width is huge (flat Gaussian).
    # If p is large negative, width is tiny (spike).
    # To avoid overflow in exp, we clip p before exp.
    
    p = params
    
    bias = p[0]
    
    y = np.full_like(x, bias, dtype=float)
    
    for i in range(3):
        center = p[1 + i*3]
        width_param = p[2 + i*3]
        amplitude = p[3 + i*3]
        
        # Ensure width is positive and stable
        # Clip width_param to [-20, 20] to avoid exp overflow/underflow
        w = np.clip(width_param, -20.0, 20.0)
        width = np.exp(w)
        
        # Gaussian: amplitude * exp(-0.5 * ((x - center) / width)^2)
        # To prevent overflow in the square if width is very small, we can clip the exponent argument
        # But if width is small, the Gaussian is a spike. 
        # Let's compute the exponent term safely.
        diff = x - center
        # If width is extremely small, diff/width is huge. 
        # We want exp(-0.5 * huge^2) -> 0.
        # We can clip the argument to the exp function.
        arg = -0.5 * (diff / width) ** 2
        
        # Clip arg to avoid underflow warnings (though exp(-1000) is just 0.0 in float64)
        # Actually, exp(-1000) is 0.0. It's safe.
        # However, if width is 0 (which it can't be due to exp), it would be inf.
        # exp(-20) is ~2e-9, exp(20) is ~4e8.
        # So width is in [2e-9, 4e8].
        # If width is 2e-9, and diff is 1, diff/width is 5e8. Squared is 2.5e17.
        # -0.5 * 2.5e17 is -1.25e17. exp(-1.25e17) is 0.0.
        # This is numerically stable in numpy.
        
        g = amplitude * np.exp(arg)
        y += g
        
    return y
# EVOLVE-BLOCK-END
