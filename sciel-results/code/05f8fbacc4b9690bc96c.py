import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [-3, 3] to t in [-1, 1]
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0(t) to T_9(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2t T_{k-1}(t) - T_{k-2}(t)
    
    # We need to return an array of predictions.
    # params has length 10.
    
    # Initialize basis
    # T_0
    basis = np.empty((10, len(x)), dtype=np.float64)
    
    t0 = np.ones_like(t)
    t1 = t.copy()
    
    basis[0] = t0
    basis[1] = t1
    
    if len(params) > 2:
        for k in range(2, 10):
            t_k = 2.0 * t * t1 - t0
            basis[k] = t_k
            t0, t1 = t1, t_k
            
    # If params has fewer than 10 elements (shouldn't happen per contract, but safe),
    # we might need to handle it, but contract says length 10.
    
    # Prediction is sum(params[k] * basis[k])
    # basis is (10, N), params is (10,)
    # We want (N,)
    y = np.dot(params, basis)
    
    return y
