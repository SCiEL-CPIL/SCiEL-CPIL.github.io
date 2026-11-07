"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [-2, 2] to t in [-1, 1] for Chebyshev stability
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_k(t) for k=0..9
    # T_0(t) = 1
    # T_1(t) = t
    # T_{k+1}(t) = 2*t*T_k(t) - T_{k-1}(t)
    
    T0 = np.ones_like(t)
    T1 = t
    
    # Precompute all basis functions
    T = [T0, T1]
    
    for k in range(2, 10):
        Tk = 2.0 * t * T[k-1] - T[k-2]
        T.append(Tk)
    
    # Linear combination of Chebyshev polynomials
    # y = sum(p_i * T_i(t))
    result = np.zeros_like(t)
    for i in range(10):
        result += params[i] * T[i]
        
    return result
# EVOLVE-BLOCK-END
