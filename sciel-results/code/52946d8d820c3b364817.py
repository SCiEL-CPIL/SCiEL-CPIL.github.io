import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a flexible model with polynomial, sigmoid, and Gaussian components.
    
    The model combines:
    - A cubic polynomial for global trend
    - Two sigmoid (tanh) units for smooth step-like transitions
    - Two Gaussian bumps for localized peaks or wells
    
    This provides a rich function space that can approximate many common smooth functions.
    
    Parameters (10 total):
    p0, p1, p2, p3: Polynomial coefficients (constant, linear, quadratic, cubic)
    p4, p5, p6: Sigmoid 1 - amplitude, slope, center
    p7, p8, p9: Sigmoid 2 - amplitude, slope, center
    
    Wait, that's 10 params but I need room for Gaussians too. Let me reconsider.
    
    Actually, with only 10 params, let's use:
    p0, p1, p2, p3: Polynomial coefficients
    p4, p5, p6: Sigmoid 1 - amplitude, slope, center  
    p7, p8, p9: Sigmoid 2 - amplitude, slope, center
    
    This gives us a polynomial plus two sigmoid units, which is quite flexible.
    """
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    
    a1 = params[4]
    c1 = params[5]
    w1 = params[6]
    
    a2 = params[7]
    c2 = params[8]
    w2 = params[9]
    
    # Polynomial part
    y = p0 + p1 * x + p2 * x * x + p3 * x * x * x
    
    # Gaussian bumps
    # Use squared width to ensure positivity and smoothness
    # w is the width parameter, larger w means wider bump
    # We use (x-c)^2 / (w^2 + eps) to avoid division by zero
    eps = 1e-6
    w1_sq = w1 * w1 + eps
    w2_sq = w2 * w2 + eps
    
    # Calculate exponents
    # Clip to avoid underflow/overflow issues in exp, though exp(-large) is just 0
    # Clipping helps if we want to maintain some gradient, but 0 is 0.
    # Standard exp is fine.
    exp1 = -((x - c1) ** 2) / w1_sq
    exp2 = -((x - c2) ** 2) / w2_sq
    
    # Clip exponents to prevent extreme values (though exp handles it, it's safer for gradients if not underflowing completely)
    # Actually, if exp underflows to 0, gradient is 0. 
    # If we clip to -100, exp(-100) is ~3e-44, gradient is tiny but non-zero.
    exp1 = np.clip(exp1, -100, 100)
    exp2 = np.clip(exp2, -100, 100)
    
    y += a1 * np.exp(exp1)
    y += a2 * np.exp(exp2)
    
    return y
