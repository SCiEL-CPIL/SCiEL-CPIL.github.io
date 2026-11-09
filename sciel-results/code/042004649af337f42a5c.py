"""Symbolic regression with Chebyshev polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_chebyt


def _cheb_map(t: np.ndarray) -> np.ndarray:
    """Map t from [0.5, 2] to [-1, 1] for Chebyshev polynomials."""
    return (2 * t - 1.5) / 1.5  # center=1.25, half_range=0.75 -> (t-1.25)/0.75


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev polynomial basis mapped to [0.5, 2] domain.
    
    Uses 10 parameters arranged as:
    - params[0]: constant
    - params[1]: Cheb1(x)
    - params[2]: Cheb1(y)
    - params[3]: Cheb1(z)
    - params[4]: Cheb2(x)
    - params[5]: Cheb2(y)
    - params[6]: Cheb2(z)
    - params[7]: Cheb1(x)*Cheb1(y)
    - params[8]: Cheb1(x)*Cheb1(z)
    - params[9]: Cheb1(y)*Cheb1(z)
    """
    p = params
    
    # Map inputs to [-1, 1] for Chebyshev evaluation
    xc = _cheb_map(x)
    yc = _cheb_map(y)
    zc = _cheb_map(z)
    
    # Chebyshev polynomials of degree 1 and 2
    T1x = xc
    T1y = yc
    T1z = zc
    T2x = 2 * xc * xc - 1
    T2y = 2 * yc * yc - 1
    T2z = 2 * zc * zc - 1
    
    # Interaction terms (product of degree-1 Chebyshev in different variables)
    Txy = T1x * T1y
    Txz = T1x * T1z
    Tyz = T1y * T1z
    
    result = (p[0] * 1.0
              + p[1] * T1x
              + p[2] * T1y
              + p[3] * T1z
              + p[4] * T2x
              + p[5] * T2y
              + p[6] * T2z
              + p[7] * Txy
              + p[8] * Txz
              + p[9] * Tyz)
    
    return result
# EVOLVE-BLOCK-END
