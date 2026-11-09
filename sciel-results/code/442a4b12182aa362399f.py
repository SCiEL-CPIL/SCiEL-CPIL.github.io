"""E. coli growth rate model: Generalized Additive Model (GAM) with Gaussian basis functions.

Structural change from hybrid quadratic+Monod model to a pure additive model.

Strategy:
- Decompose the response into smooth univariate functions of each input.
- Use 2 Gaussian basis functions per input variable (8 total) + intercept + 1 interaction.
- This creates a convex optimization landscape for BFGS (linear in parameters).
- No non-linear parameter dependencies (like Ks in Monod), so BFGS converges reliably.

Parameter allocation (10 params):
  p[0]: intercept
  p[1]: Gaussian basis 1 for b (center ~2.5, width ~1.0)
  p[2]: Gaussian basis 2 for b (center ~3.5, width ~1.0)
  p[3]: Gaussian basis 1 for s (center ~2.5, width ~1.0)
  p[4]: Gaussian basis 2 for s (center ~3.5, width ~1.0)
  p[5]: Gaussian basis 1 for temp (center ~15, width ~10)
  p[6]: Gaussian basis 2 for temp (center ~35, width ~10)
  p[7]: Gaussian basis 1 for pH (center ~5, width ~2)
  p[8]: Gaussian basis 2 for pH (center ~9, width ~2)
  p[9]: temp-pH interaction coefficient (normalized)
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate from population density, substrate, temperature and pH.
    
    GAM model: sum of Gaussian basis functions per input + one interaction term.
    
    Inputs:
        b: population density, range [1, 4.998]
        s: substrate concentration, range [1, 4.999]
        temp: temperature, range [1.002, 39.98]
        pH: pH, range [2, 12]
    
    Parameters:
        params: 10 fitted parameters
    """
    p = params
    
    # Gaussian basis functions for b (density)
    # Two bases centered in the middle of the range
    gb1 = np.exp(-((b - 2.5) ** 2) / (2 * 1.0 ** 2))
    gb2 = np.exp(-((b - 3.5) ** 2) / (2 * 1.0 ** 2))
    
    # Gaussian basis functions for s (substrate)
    gs1 = np.exp(-((s - 2.5) ** 2) / (2 * 1.0 ** 2))
    gs2 = np.exp(-((s - 3.5) ** 2) / (2 * 1.0 ** 2))
    
    # Gaussian basis functions for temp (temperature)
    # Wider bases to cover the full range
    gt1 = np.exp(-((temp - 15.0) ** 2) / (2 * 10.0 ** 2))
    gt2 = np.exp(-((temp - 35.0) ** 2) / (2 * 10.0 ** 2))
    
    # Gaussian basis functions for pH
    gp1 = np.exp(-((pH - 5.0) ** 2) / (2 * 2.0 ** 2))
    gp2 = np.exp(-((pH - 9.0) ** 2) / (2 * 2.0 ** 2))
    
    # Normalized interaction term
    T_n = (temp - 20.5) / 19.5
    pH_n = (pH - 7.0) / 5.0
    T_pH = T_n * pH_n
    
    # Combine all basis functions with their coefficients
    mu = (p[0]
          + p[1] * gb1
          + p[2] * gb2
          + p[3] * gs1
          + p[4] * gs2
          + p[5] * gt1
          + p[6] * gt2
          + p[7] * gp1
          + p[8] * gp2
          + p[9] * T_pH)
    
    return mu
# EVOLVE-BLOCK-END
