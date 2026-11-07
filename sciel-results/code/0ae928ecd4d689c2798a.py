"""Chebyshev polynomial expansion for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_chebyshev


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using Chebyshev polynomial expansion of degree 9.
    
    params[0..9] are coefficients for T_0(x) through T_9(x).
    The Chebyshev polynomials provide a stable, near-orthogonal basis on [-1, 1],
    and since x is in [-2, 2], we can still use them effectively.
    
    T_n(x) = cos(n * arccos(x)) for |x| <= 1, but scipy's eval_chebyshev
    handles x outside [-1, 1] via the polynomial definition, which is stable.
    """
    # Ensure x is a 1D array
    x = np.asarray(x, dtype=np.float64)
    
    # Compute Chebyshev polynomials T_0 through T_9
    # eval_chebyshev(n, x) computes T_n(x)
    # We can vectorize by computing each degree separately
    result = np.zeros_like(x)
    for n in range(10):
        result += params[n] * eval_chebyshev(n, x)
    
    return result
# EVOLVE-BLOCK-END
