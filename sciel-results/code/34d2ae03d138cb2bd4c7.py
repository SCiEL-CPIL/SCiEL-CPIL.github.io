import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a linear combination of Chebyshev polynomials.
    Chebyshev polynomials provide a well-conditioned basis for polynomial approximation
    on a finite interval, avoiding the numerical instability associated with high powers
    of x (e.g., x^9) which can cause BFGS to fail or converge poorly.
    
    The domain [0.1, 6] is mapped to [-1, 1] where Chebyshev polynomials are defined.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Map x from [0.1, 6] to t in [-1, 1]
    x_min = 0.1
    x_max = 6.0
    t = (2.0 * x - (x_min + x_max)) / (x_max - x_min)
    
    # Compute Chebyshev polynomials T_0(t) to T_9(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2t * T_{n-1}(t) - T_{n-2}(t)
    
    basis = np.zeros((len(x), 10), dtype=np.float64)
    
    basis[:, 0] = 1.0
    if len(x) > 0:
        basis[:, 1] = t
        
        for k in range(2, 10):
            basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
            
    # Prediction is the dot product
    return basis @ params
