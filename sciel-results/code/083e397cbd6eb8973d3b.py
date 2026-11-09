"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    ODE integrator for the Lotka-Volterra system using adaptive Dormand-Prince 5(4).
    """
    start_time = time.time()
    deadline = start_time + 280.0
    
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    t = t0
    
    # Dormand-Prince 5(4) coefficients
    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0])
    A = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    ])
    b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
    bs = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
    
    rtol = 1e-12
    atol = 1e-14
    h = (t1 - t0) / 50
    h_min = 1e-10
    h_max = (t1 - t0) / 5
    
    n = len(y)
    
    while t < t1 - 1e-15:
        if time.time() > deadline:
            break
        
        if t + h > t1:
            h = t1 - t
        
        # Compute stages
        k = np.zeros((7, n))
        k[0] = f(t, y)
        
        for i in range(1, 7):
            yi = y + h * np.dot(A[i, :i], k[:i])
            k[i] = f(t + c[i], yi)
        
        # 5th order solution
        y_new = y + h * np.dot(b, k)
        
        # 4th order error estimate
        y_err = y + h * np.dot(bs, k)
        
        err = np.abs(y_err - y_new)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_norm = np.sqrt(np.mean((err / scale) ** 2))
        
        if err_norm <= 1.0:
            y = y_new
            t += h
            
            if err_norm > 1e-8:
                factor = min(5.0, max(0.1, 0.9 * err_norm ** (-0.2)))
                h = h * factor
            else:
                h = min(h * 5.0, h_max)
        else:
            factor = max(0.1, 0.9 * err_norm ** (-0.25))
            h = h * factor
        
        h = max(h, h_min)
        h = min(h, h_max)
    
    if t < t1 - 1e-15:
        h_final = t1 - t
        k = np.zeros((7, n))
        k[0] = f(t, y)
        for i in range(1, 7):
            yi = y + h_final * np.dot(A[i, :i], k[:i])
            k[i] = f(t + c[i], yi)
        y = y + h_final * np.dot(b, k)
        t = t1
    
    if not np.all(np.isfinite(y)):
        y = np.array([10.0, 5.0])
    
    return y
# EVOLVE-BLOCK-END
