import numpy as np

N = 600

# EVOLVE-BLOCK-START
def construct():
    # Create a Gaussian-like step function
    # Support on [-1/4, 1/4]
    # x values are centers of intervals
    # Interval width h = 1/1200
    h = 1.0 / 1200.0
    # Centers: -1/4 + h/2 + i*h
    centers = -0.25 + h/2 + np.arange(N) * h
    
    # Gaussian with sigma chosen to fit the domain
    # Let sigma = 1/8 (so 4 sigma = 1/2, covering the domain)
    sigma = 1.0 / 8.0
    
    f = np.exp(-(centers**2) / (2 * sigma**2))
    
    # Return as list of floats
    return f.tolist()
# EVOLVE-BLOCK-END
