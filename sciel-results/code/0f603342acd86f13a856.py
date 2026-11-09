"""E. coli growth rate model: Full quadratic polynomial response surface.

Structural change from GAM with Gaussian bases to a full second-order polynomial.

Strategy:
- Use all 10 parameters for a complete quadratic model: intercept, 4 linear, 4 quadratic, 2 key interactions.
- This is linear in parameters, so BFGS finds the global optimum reliably.
- Captures curvature in each variable and the most biologically relevant interactions.
- No exponential computations, so faster evaluation per BFGS step.

Parameter allocation (10 params):
  p[0]: intercept
  p[1]: linear coefficient for b
  p[2]: linear coefficient for s
  p[3]: linear coefficient for temp
  p[4]: linear coefficient for pH
  p[5]: quadratic coefficient for b
  p[6]: quadratic coefficient for s
  p[7]: quadratic coefficient for temp
  p[8]: quadratic coefficient for pH
  p[9]: interaction coefficient for temp*pH (most biologically relevant)
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate from population density, substrate, temperature and pH.
    
    Full quadratic polynomial model with one key interaction term.
    
    Inputs:
        b: population density, range [1, 4.998]
        s: substrate concentration, range [1, 4.999]
        temp: temperature, range [1.002, 39.98]
        pH: pH, range [2, 12]
    
    Parameters:
        params: 10 fitted parameters
    """
    p = params
    
    # Normalize inputs for numerical stability
    b_n = (b - 3.0) / 2.0       # approx [-1, 1]
    s_n = (s - 3.0) / 2.0       # approx [-1, 1]
    temp_n = (temp - 20.5) / 19.5  # approx [-1, 1]
    pH_n = (pH - 7.0) / 5.0     # approx [-1, 1]
    
    # Quadratic model: intercept + linear + quadratic + one interaction
    mu = (p[0]
          + p[1] * b_n
          + p[2] * s_n
          + p[3] * temp_n
          + p[4] * pH_n
          + p[5] * b_n ** 2
          + p[6] * s_n ** 2
          + p[7] * temp_n ** 2
          + p[8] * pH_n ** 2
          + p[9] * temp_n * pH_n)
    
    return mu
# EVOLVE-BLOCK-END
