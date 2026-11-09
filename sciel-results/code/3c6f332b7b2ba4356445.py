"""
Strategy: Non-linear parametric model with adaptive structure.
Unlike the fixed Chebyshev basis (linear in params), this model uses parameters
to define the shape of the function directly. We use a hybrid of a low-order
polynomial (for global trend) and a sum of Gaussian bumps (for local features).
This is non-linear in parameters, but the landscape is often smoother for BFGS
if the hidden function is smooth, as it can "find" the bumps rather than
"approximate" them with high-degree polynomials which can oscillate (Runge's phenomenon).

Model:
y(x) = p0 + p1*x + p2*x^2 + sum_{i=0}^{3} p_{3+i} * exp(-0.5 * ((x - c_i)/w_i)^2)
Where:
c_i = p_{7+i} (centers)
w_i = max(0.1, p_{11+i}) -- wait, we only have 10 params.

Let's fit 10 params:
p0, p1, p2: Polynomial coefficients (1, x, x^2)
p3, p4: Center and Width of Gaussian 1
p5, p6: Center and Width of Gaussian 2
p7, p8: Center and Width of Gaussian 3
p9: Amplitude? No, we need amplitudes for the Gaussians too.

Revised Model (10 params):
y(x) = p0 * (1 + p1*x + p2*x^2 + p3*x^3)  # Cubic polynomial with global scale
       + p4 * exp(-0.5 * ((x - p5)/p6)^2)  # Gaussian 1
       + p7 * exp(-0.5 * ((x - p8)/p9)^2)  # Gaussian 2
Wait, that's 10 params: p0..p9.
p0: Scale for poly
p1, p2, p3: Poly coeffs
p4: Amp G1
p5: Center G1
p6: Width G1
p7: Amp G2
p8: Center G2
p9: Width G2

This allows the model to capture smooth global trends plus two local bumps.
If the function is just a polynomial, the Gaussians can shrink (width -> 0 or amp -> 0).
If the function has bumps, the poly can be flat.

To avoid BFGS getting stuck in local minima due to width parameters being near 0 (singular),
we clamp widths to a minimum value, e.g., 0.1.
We also need to ensure numerical stability.
"""

import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Non-linear parametric model:
    y = p0 * (1 + p1*x + p2*x^2 + p3*x^3) 
      + p4 * exp(-0.5 * ((x - p5)/w1)^2)
      + p7 * exp(-0.5 * ((x - p8)/w2)^2)
    
    Where w1 = max(0.1, |p6|) and w2 = max(0.1, |p9|)
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(params, dtype=float)
    
    # Safety check
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(p)):
        return np.zeros_like(x)
        
    # Extract parameters
    p0, p1, p2, p3 = p[0], p[1], p[2], p[3]
    a1, c1, w1_raw = p[4], p[5], p[6]
    a2, c2, w2_raw = p[7], p[8], p[9]
    
    # Clamp widths to prevent division by zero and extreme peaks
    w1 = np.maximum(0.1, np.abs(w1_raw))
    w2 = np.maximum(0.1, np.abs(w2_raw))
    
    # Compute polynomial part
    # Using Horner's method for stability: 1 + x*(p1 + x*(p2 + x*p3))
    poly = 1.0 + x * (p1 + x * (p2 + x * p3))
    y_poly = p0 * poly
    
    # Compute Gaussian parts
    # exp(-0.5 * ((x - c) / w)^2)
    # To avoid overflow in exp, we can clip the argument, but with w>=0.1 and x in [-2,2], 
    # (x-c) is at most 4, so ((x-c)/w)^2 is at most (4/0.1)^2 = 1600. 
    # exp(-0.5 * 1600) is 0.0. No overflow risk, just underflow to 0 which is fine.
    
    z1 = (x - c1) / w1
    g1 = np.exp(-0.5 * z1 * z1)
    
    z2 = (x - c2) / w2
    g2 = np.exp(-0.5 * z2 * z2)
    
    y_gauss = a1 * g1 + a2 * g2
    
    y = y_poly + y_gauss
    
    # Ensure finite output
    if not np.all(np.isfinite(y)):
        y = np.where(np.isfinite(y), y, 0.0)
        
    return y
