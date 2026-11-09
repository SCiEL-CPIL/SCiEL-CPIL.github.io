"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: Chebyshev Polynomial Basis (Degree 9).
    
    y(x) = sum_{k=0}^{9} p[k] * T_k(x)
    
    This model uses Chebyshev polynomials of the first kind, which are orthogonal
    on [-1, 1] with respect to the weight function 1/sqrt(1-x^2). Since the input
    x is drawn from [-2, 2], we map it to the standard domain [-1, 1] via
    t = x / 2.
    
    Chebyshev polynomials provide excellent approximation properties for smooth
    functions on compact intervals, minimizing the maximum error (minimax property).
    They avoid the Runge phenomenon associated with high-degree monomial bases
    and are generally more numerically stable than monomials x^k for higher degrees.
    
    Parameters:
    p[0] ... p[9]: Coefficients for T_0(t) ... T_9(t)
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Map x from [-2, 2] to [-1, 1]
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 ... T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_{k+1}(t) = 2*t*T_k(t) - T_{k-1}(t)
    
    T_prev = np.ones_like(t)  # T_0
    T_curr = t.copy()          # T_1
    
    result = p[0] * T_prev
    if len(p) > 1:
        result = result + p[1] * T_curr
    
    for k in range(2, 10):
        T_next = 2.0 * t * T_curr - T_prev
        result = result + p[k] * T_next
        T_prev = T_curr
        T_curr = T_next
        
    return result
# EVOLVE-BLOCK-END
