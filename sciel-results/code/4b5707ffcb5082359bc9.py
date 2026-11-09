import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x to [-1, 1] for Chebyshev stability
    z = 0.5 * x
    
    # T_0
    t0 = np.ones_like(z)
    # T_1
    t1 = z.copy()
    
    # Accumulate result
    # params[0] * T_0 + params[1] * T_1 + ...
    result = params[0] * t0 + params[1] * t1
    
    if len(params) > 2:
        # T_k = 2z T_{k-1} - T_{k-2}
        t_prev = t0
        t_curr = t1
        for k in range(2, 10):
            t_next = 2 * z * t_curr - t_prev
            result += params[k] * t_next
            t_prev = t_curr
            t_curr = t_next
            
    return result
