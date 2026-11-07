import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [-2, 2] to t in [-1, 1]
    t = x * 0.5
    
    # Compute Chebyshev polynomials T_0(t) ... T_9(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    n_params = 10
    basis = np.zeros((len(x), n_params))
    
    if n_params > 0:
        basis[:, 0] = 1.0
    if n_params > 1:
        basis[:, 1] = t
        
    for k in range(2, n_params):
        basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
        
    # Linear combination
    return basis @ params
