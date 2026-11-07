"""Symbolic regression with 1 input(s)."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a Chebyshev polynomial basis.
    Maps x from [-3, 3] to t in [-1, 1], then uses T_0(t), T_1(t), ..., T_9(t).
    This is a linear model: f(x) = sum_{k=0}^{9} c_k T_k(x/3)
    where params = [c_0, c_1, ..., c_9].
    Chebyshev polynomials are orthogonal on [-1,1] and well-conditioned.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Map x from [-3, 3] to t in [-1, 1]
    t = x / 3.0
    
    # Construct Chebyshev basis matrix using recurrence relation
    # T_0(t) = 1, T_1(t) = t, T_{k+1}(t) = 2*t*T_k(t) - T_{k-1}(t)
    basis_matrix = np.empty((n, 10), dtype=np.float64)
    basis_matrix[:, 0] = 1.0  # T_0
    if n > 0:
        basis_matrix[:, 1] = t  # T_1
    
    for k in range(2, 10):
        # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
        basis_matrix[:, k] = 2.0 * t * basis_matrix[:, k-1] - basis_matrix[:, k-2]
    
    # Linear combination
    result = basis_matrix @ params
    
    return result
# EVOLVE-BLOCK-END
