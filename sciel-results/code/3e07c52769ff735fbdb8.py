import numpy as np
from scipy.optimize import minimize
import time
import math

_DEADLINE = 205.0
_START_TIME = None

def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    degrees = [0, 4, 8, 12]
    
    # Weights w_k = 2^k * k!
    w = np.array([2**d * math.factorial(d) for d in degrees], dtype=float)
    
    A_diag = np.array([(d + 0.5) * wk for d, wk in zip(degrees, w)], dtype=float)
    B_diag = np.array([2 * d * wk for d, wk in zip(degrees, w)], dtype=float)
    
    K = 103968.0
    w0, w4, w8, w12 = w
    A0, A4, A8, A12 = A_diag
    B0, B4, B8, B12 = B_diag
    
    alpha = np.array([1.0, -12.0, 1008.0]) / K
    
    W_0 = np.diag([w0, w4, w8])
    A_0 = np.diag([A0, A4, A8])
    B_0 = np.diag([B0, B4, B8])
    
    def objective(v):
        c12 = alpha @ v
        N = v @ W_0 @ v + c12**2 * w12
        A = v @ A_0 @ v + c12**2 * A12
        B = v @ B_0 @ v + c12**2 * B12
        if N <= 0 or A <= 0 or B <= 0:
            return 1e100
        return np.sqrt(A * B) / N
    
    v0 = np.array([1.0, 0.0, 0.0])
    best_v = v0.copy()
    best_val = objective(v0)
    
    np.random.seed(42)
    for i in range(100):
        if _time_left() < 10:
            break
        v_init = np.random.randn(3) * 0.5 + v0
        res = minimize(objective, v_init, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
        if res.fun < best_val:
            best_val = res.fun
            best_v = res.x
        
    if _time_left() > 5:
        res = minimize(objective, best_v, method='L-BFGS-B', 
                       options={'maxiter': 1000, 'gtol': 1e-14})
        if res.fun < best_val:
            best_val = res.fun
            best_v = res.x
            
    return np.array(best_v)
