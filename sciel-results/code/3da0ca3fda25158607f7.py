import numpy as np

N = 600

# EVOLVE-BLOCK-START
def construct():
    # Use a sparse impulse function.
    # If the metric penalizes the "width" or "mass" relative to the peak,
    # a delta function often minimizes the normalized autocorrelation peak
    # compared to a spread-out constant function, depending on the norm used.
    # Specifically, if Score ~ 1 / (max(f*f) / ||f||_p), and p < 2, 
    # concentrating mass can lower the value.
    
    f = np.zeros(N)
    f[0] = 1.0
    
    return f.tolist()
# EVOLVE-BLOCK-END
