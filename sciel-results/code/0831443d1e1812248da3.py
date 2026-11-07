import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Chebyshev Polynomial approach
    # Map x in [-2, 2] to t in [-1, 1]
    t = x / 2.0
    
    n = len(x)
    basis = np.zeros((n, 10))
    
    # T_0(t) = 1
    basis[:, 0] = 1.0
    
    # T_1(t) = t
    basis[:, 1] = t
    
    # Recurrence: T_{k+1}(t) = 2*t*T_k(t) - T_{k-1}(t)
    if n > 0:
        T_prev = basis[:, 0].copy() # T_0
        T_curr = basis[:, 1].copy() # T_1
        
        for k in range(2, 10):
            T_next = 2.0 * t * T_curr - T_prev
            basis[:, k] = T_next
            T_prev = T_curr
            T_curr = T_next
            
    return basis @ params
