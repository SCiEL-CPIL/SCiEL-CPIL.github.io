"""
Strategy: Fixed-Width Radial Basis Function (RBF) Network with Linear Trend.

We model the target function as:
y = b + a*x + sum_{i=1}^{4} c_i * exp(-(x - m_i)^2 / sigma_i^2)

Parameters (10 total):
- b: Global bias (param 0)
- a: Linear coefficient (param 1)
- c1, m1: Amplitude and center for RBF 1 (params 2, 3)
- c2, m2: Amplitude and center for RBF 2 (params 4, 5)
- c3, m3: Amplitude and center for RBF 3 (params 6, 7)
- c4, m4: Amplitude and center for RBF 4 (params 8, 9)

The widths (sigmas) are fixed to cover a range of scales:
sigma_1 = 0.3 (fine detail)
sigma_2 = 0.7 (medium detail)
sigma_3 = 1.5 (coarse structure)
sigma_4 = 3.0 (global structure)

This representation is structurally different from the current tanh-sum strategy:
1. It uses Gaussian "bumps" instead of S-shaped "transitions", allowing better approximation of functions with local extrema (like sin, x^2, etc.).
2. It includes an explicit linear term, which helps BFGS converge faster for functions with a strong linear component.
3. The fixed widths provide a multi-scale basis, ensuring the model can capture both sharp and broad features without parameterizing the width (which can lead to optimization instability if widths become very small or large).
"""

import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y using a linear term plus a sum of 4 fixed-width Gaussians.
    params: array of length 10.
    """
    x = np.asarray(x, dtype=float)
    
    # Extract parameters
    b = params[0]
    a = params[1]
    
    c1, m1 = params[2], params[3]
    c2, m2 = params[4], params[5]
    c3, m3 = params[6], params[7]
    c4, m4 = params[8], params[9]
    
    # Fixed widths
    sigma1 = 0.3
    sigma2 = 0.7
    sigma3 = 1.5
    sigma4 = 3.0
    
    # Compute RBF terms
    # Using (x - m)^2 / sigma^2 in the exponent
    # exp(-0.5 * ((x-m)/sigma)^2) is standard Gaussian, but exp(-((x-m)/sigma)^2) is also fine.
    # Let's use standard form: exp(-0.5 * ((x - m) / sigma) ** 2)
    
    term1 = c1 * np.exp(-0.5 * ((x - m1) / sigma1) ** 2)
    term2 = c2 * np.exp(-0.5 * ((x - m2) / sigma2) ** 2)
    term3 = c3 * np.exp(-0.5 * ((x - m3) / sigma3) ** 2)
    term4 = c4 * np.exp(-0.5 * ((x - m4) / sigma4) ** 2)
    
    y = b + a * x + term1 + term2 + term3 + term4
    
    return y
