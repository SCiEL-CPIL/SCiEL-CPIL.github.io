import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure x is 1D array
    x = np.asarray(x).flatten()
    
    # Scale input x from [-2, 2] to t in [-1, 1]
    t = x / 2.0
    
    # Compute Chebyshev polynomials of the first kind up to degree 9
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n+1}(t) = 2*t*T_n(t) - T_{n-1}(t)
    
    n_points = len(t)
    basis = np.zeros((n_points, 10))
    basis[:, 0] = 1.0
    basis[:, 1] = t
    
    for k in range(2, 10):
        basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
        
    # params is (10,)
    # result is (N,)
    result = basis @ params
    
    return result
