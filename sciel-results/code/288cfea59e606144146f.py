import numpy as np
import time

def equation(x, y, params):
    # Use a sparse linear combination of 10 Radial Basis Functions (RBFs)
    # with centers and widths fixed by a deterministic grid over [-1, 1].
    # This allows the optimizer to fit only the amplitudes to capture
    # localized, non-polynomial features.
    
    # Define 10 fixed centers on a 3x3 grid plus one at origin
    # Centers: (cx, cy)
    centers = [
        (-1.0, -1.0), (-1.0, 0.0), (-1.0, 1.0),
        (0.0, -1.0), (0.0, 0.0), (0.0, 1.0),
        (1.0, -1.0), (1.0, 0.0), (1.0, 1.0),
        (0.0, 0.0)  # Duplicate center with different width for flexibility
    ]
    
    # Fixed widths (sigma) for each RBF
    # Smaller sigma = more localized, larger sigma = smoother
    widths = [
        0.3, 0.5, 0.3,
        0.5, 0.7, 0.5,
        0.3, 0.5, 0.3,
        1.0
    ]
    
    # Compute RBF values: exp(-||x - c||^2 / (2 * sigma^2))
    result = np.zeros_like(x, dtype=np.float64)
    
    for i in range(10):
        cx, cy = centers[i]
        sigma = widths[i]
        
        # Squared distance from each point to center
        dx = x - cx
        dy = y - cy
        dist_sq = dx**2 + dy**2
        
        # Gaussian RBF
        rbf = np.exp(-dist_sq / (2.0 * sigma**2))
        
        result += params[i] * rbf
    
    return result
