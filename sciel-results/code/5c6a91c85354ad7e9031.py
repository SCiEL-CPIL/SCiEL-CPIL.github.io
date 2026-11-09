"""ODE solver: Heat2D 10x10 – integrate(f, t_span, y0)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Adaptive RK45 (Dormand-Prince) solver with error control.
    
    Returns the final state y(t_end) for the IVP y' = f(t, y).
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Dormand-Prince coefficients (standard)
    A = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
    B = np.array([0, 0, 0, 0, 0, 0, 0,
                  1/5, 0, 0, 0, 0, 0,
                  3/40, 9/40, 0, 0, 0, 0,
                  44/45, -56/15, 32/9, 0, 0, 0,
                  19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0,
                  9017/3168, -355/33, 46732/5247, 49/176, 51031/18656, 0,
                  35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
    C5 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
    C4 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
    
    # Initial step size
    f0 = f(t0, y)
    calls = 1
    scale = np.maximum(1e-6, np.abs(y))
    delta1 = scale / np.maximum(1e-30, np.abs(f0))
    h = np.min(delta1)
    h = min(max(h, 1e-10), (t1 - t0) / 10)
    
    atol = 1e-10
    rtol = 1e-8
    deadline = time.time() + 280
    t = t0
    max_calls = 2000000
    
    while t < t1:
        if time.time() > deadline - 5 or calls >= max_calls:
            if t1 - t > 1e-15:
                h_rem = t1 - t
                fy = f(t, y)
                calls += 1
                y = y + h_rem * fy
                t = t1
            break
        
        h = min(h, t1 - t)
        
        # Compute stages
        k1 = f(t, y)
        calls += 1
        
        y2 = y + h * B[8] * k1
        k2 = f(t + A[1]*h, y2)
        calls += 1
        
        y3 = y + h * (B[16]*k1 + B[17]*k2)
        k3 = f(t + A[2]*h, y3)
        calls += 1
        
        y4 = y + h * (B[24]*k1 + B[25]*k2 + B[26]*k3)
        k4 = f(t + A[3]*h, y4)
        calls += 1
        
        y5 = y + h * (B[32]*k1 + B[33]*k2 + B[34]*k3 + B[35]*k4)
        k5 = f(t + A[4]*h, y5)
        calls += 1
        
        y6 = y + h * (B[40]*k1 + B[41]*k2 + B[42]*k3 + B[43]*k4 + B[44]*k5)
        k6 = f(t + A[5]*h, y6)
        calls += 1
        
        # 5th and 4th order
        y5th = y + h * (C5[0]*k1 + C5[2]*k3 + C5[3]*k4 + C5[4]*k5 + C5[5]*k6)
        y4th = y + h * (C4[0]*k1 + C4[2]*k3 + C4[3]*k4 + C4[4]*k5 + C4[5]*k6 + C4[6]*k6)
        
        err = y5th - y4th
        err_norm = np.max(np.abs(err) / (atol + rtol * np.maximum(np.abs(y), np.abs(y5th))))
        
        if err_norm <= 1.0:
            y = y5th
            t += h
            if err_norm > 0:
                h = h * min(5.0, max(0.2, 0.9 * err_norm ** (-0.2)))
            else:
                h = h * 5.0
        else:
            h = h * max(0.1, 0.9 * err_norm ** (-0.25))
        
        h = min(max(h, 1e-12), (t1 - t) if t < t1 else h)
    
    return y
# EVOLVE-BLOCK-END
