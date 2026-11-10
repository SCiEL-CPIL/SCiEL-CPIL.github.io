# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale input to [-1, 1] for numerical stability
    t = x / 3.0

    # Use a Chebyshev-like polynomial basis up to degree 9
    # This gives better conditioning than monomial basis
    # Chebyshev polynomials: T_0=1, T_1=t, T_n=2*t*T_{n-1} - T_{n-2}
    
    # Precompute Chebyshev basis
    # T_0
    T0 = np.ones_like(t)
    # T_1
    T1 = t.copy()
    
    basis = [T0, T1]
    
    if len(params) > 2:
        for k in range(2, 10):
            Tk = 2.0 * t * basis[-1] - basis[-2]
            basis.append(Tk)
    
    # Stack into matrix (N, 10)
    basis_matrix = np.column_stack(basis)
    
    # Linear combination
    y = basis_matrix @ params
    
    return y
# EVOLVE-BLOCK-END
