"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.special import roots_laguerre

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the 12 positive double-root locations using Chebyshev nodes.
    This approach uses the roots of the Chebyshev polynomial of the first kind,
    mapped to a positive interval, which often provides optimal distribution
    for polynomial-based uncertainty bounds compared to Laguerre roots.
    """
    m = 12
    # Define the interval for the roots.
    # The seed used 4 to 48. We use a similar range [1, 50] to ensure positivity
    # and spread, which might be favorable for the C4 bound if it penalizes
    # clustering near zero (as Laguerre roots do).
    a = 1.0
    b = 50.0
    
    # Chebyshev nodes of the first kind in [a, b]
    # x_k = (a+b)/2 + (b-a)/2 * cos((2k+1)pi / (2m))
    k = np.arange(m)
    theta = (2 * k + 1) * np.pi / (2 * m)
    nodes = (a + b) / 2.0 + (b - a) / 2.0 * np.cos(theta)
    
    # Ensure they are sorted in ascending order and positive
    nodes = np.sort(nodes)
    
    # Convert to list of floats
    return [float(x) for x in nodes]
# EVOLVE-BLOCK-END
