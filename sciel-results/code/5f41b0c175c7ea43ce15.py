"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Chebyshev polynomial basis.
    
    The model is: y = sum_{i=0}^{9} p_i * T_i(x/3)
    where T_i are Chebyshev polynomials of the first kind.
    This maps the input domain [-3, 3] to [-1, 1] to ensure stability 
    and orthogonality of the basis functions.
    """
    # Map x from [-3, 3] to [-1, 1]
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0(t) to T_9(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    # Initialize features array: shape (N, 10)
    N = len(x)
    features = np.zeros((N, 10))
    
    # T_0
    features[:, 0] = 1.0
    
    # T_1
    features[:, 1] = t
    
    # T_2 to T_9
    for i in range(2, 10):
        features[:, i] = 2.0 * t * features[:, i-1] - features[:, i-2]
    
    # Linear combination of features with parameters
    return features @ params
# EVOLVE-BLOCK-END
