import numpy as np
from scipy.special import eval_chebyt

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [-2, 2] to [-1, 1]
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    # We can build the basis matrix
    # But we need to return the prediction for each x.
    # Since we have 10 params, we sum p_i * T_i(t)
    
    # Efficient way:
    # p[0] * T_0 + p[1] * T_1 + ...
    
    # Let's compute T_0 and T_1
    T0 = np.ones_like(t)
    T1 = t.copy()
    
    pred = params[0] * T0 + params[1] * T1
    
    if len(params) > 2:
        T_prev = T0
        T_curr = T1
        for i in range(2, 10):
            T_next = 2.0 * t * T_curr - T_prev
            pred += params[i] * T_next
            T_prev = T_curr
            T_curr = T_next
            
    return pred
