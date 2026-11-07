"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y from x using a linear combination of 10 Chebyshev polynomials of the first kind.
    Chebyshev polynomials are orthogonal on [-1, 1] and provide a numerically stable basis
    for approximating smooth functions, avoiding the ill-conditioning issues of raw polynomials.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)), mode='constant')
    
    # Scale x from [-3, 3] to [-1, 1] for Chebyshev stability
    x_scaled = x / 3.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(x) = 1
    # T_1(x) = x
    # T_{n+1}(x) = 2x*T_n(x) - T_{n-1}(x)
    
    basis = np.zeros((10, len(x)), dtype=np.float64)
    
    if len(x) > 0:
        basis[0, :] = 1.0
        if len(x) > 0:
            basis[1, :] = x_scaled
        
        for n in range(2, 10):
            basis[n, :] = 2.0 * x_scaled * basis[n-1, :] - basis[n-2, :]
    
    # Linear combination
    # y = sum(p[i] * T_i(x))
    y = np.dot(p, basis)
    
    return y

# EVOLVE-BLOCK-END
