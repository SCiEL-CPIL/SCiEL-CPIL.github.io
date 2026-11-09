"""
Strategy: Non-linear parametric model with adaptive structure.
Instead of a fixed linear basis (like Chebyshev), we define a model where the 10 parameters
control both the coefficients and the structural features (centers, widths, frequencies).
This allows the model to represent a much wider class of functions (exponentials, sines,
shifted Gaussians) that a fixed polynomial basis cannot approximate well without
numerical instability.

Model:
y(x) = p0 + p1*x + p2*exp(p3*x) + p4*exp(-p5*x^2) + p6*sin(p7*x) + p8*cos(p9*x)

This uses all 10 parameters non-linearly.
- p0, p1: Linear baseline.
- p2, p3: Exponential term (can model growth/decay).
- p4, p5: Gaussian term (can model bumps/peaks).
- p6, p7: Sine term (can model oscillations).
- p8, p9: Cosine term (phase-shifted oscillations).

BFGS will optimize these 10 parameters. The non-linearity in p3, p5, p7, p9 allows
the optimizer to "move" the features of the function, not just scale them.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Non-linear parametric model.
    params: array of length 10.
    p[0]: offset
    p[1]: linear coeff
    p[2]: exp coeff, p[3]: exp rate
    p[4]: gauss coeff, p[5]: gauss width
    p[6]: sine coeff, p[7]: sine freq
    p[8]: cosine coeff, p[9]: cosine freq
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(params, dtype=float)
    
    # Safety check
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(p)):
        return np.zeros_like(x)
        
    # Extract parameters
    c0 = p[0]
    c1 = p[1]
    
    c2 = p[2]
    r3 = p[3]
    
    c4 = p[4]
    w5 = p[5]
    
    c6 = p[6]
    f7 = p[7]
    
    c8 = p[8]
    f9 = p[9]
    
    # Compute terms with numerical stability guards
    # 1. Linear part
    y = c0 + c1 * x
    
    # 2. Exponential part: c2 * exp(r3 * x)
    # Clip exponent to avoid overflow
    exp_arg = r3 * x
    exp_arg = np.clip(exp_arg, -50, 50)
    y += c2 * np.exp(exp_arg)
    
    # 3. Gaussian part: c4 * exp(-w5 * x^2)
    # Clip w5 to avoid negative width (which would be exp(+|w|x^2) -> overflow)
    # If w5 is negative, it acts like an inverse Gaussian, which might be needed?
    # Usually width is positive. Let's allow negative but clip the exponent.
    gauss_arg = -w5 * (x ** 2)
    gauss_arg = np.clip(gauss_arg, -50, 50)
    y += c4 * np.exp(gauss_arg)
    
    # 4. Sine part: c6 * sin(f7 * x)
    y += c6 * np.sin(f7 * x)
    
    # 5. Cosine part: c8 * cos(f9 * x)
    y += c8 * np.cos(f9 * x)
    
    # Ensure finite output
    if not np.all(np.isfinite(y)):
        y = np.where(np.isfinite(y), y, 0.0)
        
    return y
# EVOLVE-BLOCK-END
