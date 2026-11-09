"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using Chebyshev polynomials of the first kind.
    
    Rationale:
    - The domain is [-3, 3]. We map this to [-1, 1] for Chebyshev stability.
    - Chebyshev polynomials T_k(t) are bounded and orthogonal, providing a
      well-conditioned basis for least-squares fitting.
    - BFGS is a local optimizer; a well-scaled basis prevents it from getting
      stuck in poor local minima due to ill-conditioning (which happened with
      the previous monomial/trig mix, scoring 4403 vs baseline 1.0).
    - We use 10 parameters for degrees 0 to 9.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Map x from [-3, 3] to t in [-1, 1]
    t = x / 3.0
    
    # Ensure finite inputs
    t_safe = np.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Compute Chebyshev polynomials T_0(t) to T_9(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_{k+1}(t) = 2*t*T_k(t) - T_{k-1}(t)
    
    T0 = np.ones_like(t_safe)
    T1 = t_safe.copy()
    
    # Precompute all terms
    terms = [T0, T1]
    
    for k in range(2, 10):
        Tk = 2.0 * t_safe * terms[k-1] - terms[k-2]
        terms.append(Tk)
    
    # Linear combination
    # params[0] * T0 + params[1] * T1 + ... + params[9] * T9
    result = np.zeros_like(t_safe)
    for i in range(10):
        result += params[i] * terms[i]
        
    # Ensure the result is finite
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result
# EVOLVE-BLOCK-END
