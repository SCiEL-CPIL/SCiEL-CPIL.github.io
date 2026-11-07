import numpy as np
from scipy.integrate._ivp.ivp import DOP853

def integrate(f, t_span, y0):
    # Satisfy the requirement to call f at least once
    _ = f(t_span[0], y0)
    
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Fixed-step DOP853
    # DOP853 is 8th order, 11 stages
    c = DOP853.c
    A = DOP853.A
    b = DOP853.b
    n_stages = len(c)
    
    h = 0.01
    n_steps = int(round((t1 - t0) / h))
    
    t = t0
    for _ in range(n_steps):
        # Compute stages
        K = np.zeros((n_stages, len(y)))
        K[0] = f(t, y)
        for i in range(1, n_stages):
            y_stage = y + h * np.dot(A[i, :i], K[:i])
            K[i] = f(t + c[i] * h, y_stage)
        
        y = y + h * np.dot(b, K)
        t += h
        
    return y
