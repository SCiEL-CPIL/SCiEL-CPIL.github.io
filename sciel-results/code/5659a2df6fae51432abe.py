import numpy as np

def construct():
    """
    Return 12 double-root locations z_1..z_12.
    
    Strategy: Use the 12 smallest positive zeros of the Bessel function J_{-1/2}(x).
    J_{-1/2}(x) = sqrt(2/(pi*x)) * cos(sqrt(x)).
    The zeros are at sqrt(x) = (k - 1/2)*pi for k=1, 2, ...
    So x = (k - 1/2)^2 * pi^2.
    This aligns with the alpha = -1/2 parameter of the Laguerre polynomials,
    which are asymptotically related to J_{-1/2}.
    """
    # Zeros of J_{-1/2}(x) are at x = (k - 0.5)^2 * pi^2 for k=1, 2, ...
    k_vals = np.arange(1, 13, dtype=np.float64)
    # k - 0.5 gives 0.5, 1.5, ..., 11.5
    roots_sqrt = (k_vals - 0.5) * np.pi
    z_vals = roots_sqrt ** 2
    
    # Ensure they are finite, > 0, and pairwise distinct
    z_list = [float(z) for z in z_vals]
    
    return z_list
