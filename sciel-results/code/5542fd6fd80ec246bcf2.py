"""Symbolic regression with 3 inputs - Gaussian RBF + polynomial hybrid model."""
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the hidden function using a hybrid model:
    one Gaussian radial basis function + linear and interaction terms.
    
    Structure:
    G = A * exp(-((x-cx)^2 + (y-cy)^2 + (z-cz)^2) / (2*sigma^2))
    pred = G + a*x + b*y + c*z + d + e*(x*y + y*z + z*x)
    
    The Gaussian captures local non-linear structure while the polynomial
    base handles global trends. This is structurally different from tanh-based
    or pure polynomial models.
    """
    p = params.flatten()
    
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size), mode='constant')
    else:
        p = p[:10]
        
    cx, cy, cz, sigma, A, a, b, c, d, e = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]
    
    # Ensure sigma is positive and reasonable
    sigma_safe = np.exp(np.clip(sigma, -10, 10))
    
    # Gaussian radial basis function
    r2 = (x - cx)**2 + (y - cy)**2 + (z - cz)**2
    G = A * np.exp(-r2 / (2.0 * sigma_safe**2))
    
    # Linear + interaction terms
    interaction = x * y + y * z + z * x
    pred = G + a * x + b * y + c * z + d + e * interaction
    
    return pred
