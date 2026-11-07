"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 50

# EVOLVE-BLOCK-START
def construct():
    # Use a flat-top cosine window to maximize the flatness of the autocorrelation near the peak.
    # A flatter autocorrelation g near k=0 increases the terms y[i]^2 + y[i]y[i+1] + y[i+1]^2
    # because y[i] and y[i+1] are closer in magnitude.
    
    # Define the window shape
    # t is in [0, 1]
    t = np.linspace(0, 1, N)
    
    # Flat-top region: |t - 0.5| < 0.25  (i.e., 0.25 < t < 0.75)
    # Taper region: 0 <= t <= 0.25 and 0.75 <= t <= 1
    
    h = np.ones(N)
    
    # Left taper
    mask_left = t <= 0.25
    if np.any(mask_left):
        # Map t from [0, 0.25] to [0, 1] for the cosine taper
        x = t[mask_left] / 0.25
        # Cosine taper: 0.5 * (1 - cos(pi * x)) goes from 0 to 1
        h[mask_left] = 0.5 * (1 - np.cos(np.pi * x))
        
    # Right taper
    mask_right = t >= 0.75
    if np.any(mask_right):
        # Map t from [0.75, 1] to [0, 1] for the cosine taper
        x = (t[mask_right] - 0.75) / 0.25
        # Cosine taper: 0.5 * (1 + cos(pi * x)) goes from 1 to 0
        h[mask_right] = 0.5 * (1 + np.cos(np.pi * x))
        
    # Ensure nonnegative and finite
    h = np.maximum(h, 0.0)
    
    return list(h)
# EVOLVE-BLOCK-END
