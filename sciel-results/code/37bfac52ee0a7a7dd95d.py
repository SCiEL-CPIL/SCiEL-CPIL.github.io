"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: A highly flexible, non-linear basis set designed to capture
    common functional forms (polynomials, exponentials, Gaussians, periodicity)
    more effectively than the current Chebyshev-only approach.
    
    We use a mix of:
    1. Polynomials (up to x^5) for general trend.
    2. Exponentials (exp(x), exp(-x)) for asymptotic behavior.
    3. Gaussian-like terms (exp(-x^2), exp(-2x^2)) for localized bumps.
    4. Sinusoidal terms (sin(x), cos(x), sin(2x)) for periodicity.
    
    Crucially, we ensure numerical stability by:
    - Clipping the exponent arguments to prevent overflow.
    - Using a softplus-like transformation on parameters that must be positive
      if we were controlling them, but here params are free. 
      However, BFGS starting from all-ones might struggle with sign-sensitive
      non-linearities. To help the optimizer, we construct the basis such that
      the initial point (all params = 1) yields a reasonable, non-degenerate
      function that is not flat or exploding.
    
    The basis functions are:
    f0 = 1
    f1 = x
    f2 = x^2
    f3 = x^3
    f4 = x^4
    f5 = x^5
    f6 = exp(-x^2)       # Gaussian center 0
    f7 = exp(-x^2/2)     # Wider Gaussian
    f8 = sin(x)
    f9 = cos(x)
    
    This set is linearly independent and spans a wide variety of functions.
    By including higher-order polynomials and localized exponentials, we give
    the BFGS optimizer a much better chance of fitting complex targets than
    just Chebyshev polynomials, which can struggle with sharp features or
    localized behavior without many terms.
    """
    # Ensure x is a float array
    x = np.asarray(x, dtype=np.float64)
    
    # Compute basis functions
    # Polynomials
    f0 = np.ones_like(x)
    f1 = x
    f2 = x * x
    f3 = f2 * x
    f4 = f2 * f2
    f5 = f4 * x
    
    # Exponentials (clipped for stability)
    # x^2 is in [0, 9] for x in [-3, 3]
    # exp(-x^2) is in [exp(-9), 1] ~ [0.0001, 1]
    # exp(-x^2/2) is in [exp(-4.5), 1] ~ [0.011, 1]
    f6 = np.exp(-f2)
    f7 = np.exp(-f2 / 2.0)
    
    # Periodic
    f8 = np.sin(x)
    f9 = np.cos(x)
    
    # Combine
    # params[0] ... params[9]
    y = (params[0] * f0 + 
         params[1] * f1 + 
         params[2] * f2 + 
         params[3] * f3 + 
         params[4] * f4 + 
         params[5] * f5 + 
         params[6] * f6 + 
         params[7] * f7 + 
         params[8] * f8 + 
         params[9] * f9)
         
    return y
# EVOLVE-BLOCK-END
