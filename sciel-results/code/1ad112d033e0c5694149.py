"""
Symbolic regression with 1 input.
Uses Chebyshev polynomial basis of degree 9 for better conditioning.
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluates a Chebyshev polynomial series of degree 9.
    
    The model is:
        f(x) = sum_{i=0}^{9} params[i] * T_i(x/3.0)
    
    Chebyshev polynomials T_i are orthogonal on [-1, 1] and bounded by 1,
    which provides excellent numerical conditioning for BFGS optimization.
    """
    # Scale input to [-1, 1]
    z = x / 3.0
    
    # Compute Chebyshev polynomials T_0(z) through T_9(z)
    # T_0(z) = 1
    # T_1(z) = z
    # T_{n}(z) = 2*z*T_{n-1}(z) - T_{n-2}(z)
    N = 10
    cheb = np.zeros((len(z), N))
    cheb[:, 0] = 1.0
    if N > 1:
        cheb[:, 1] = z
        for k in range(2, N):
            cheb[:, k] = 2.0 * z * cheb[:, k-1] - cheb[:, k-2]
    
    # Linear combination of Chebyshev basis functions
    y = cheb @ params
    
    return y
# EVOLVE-BLOCK-END
