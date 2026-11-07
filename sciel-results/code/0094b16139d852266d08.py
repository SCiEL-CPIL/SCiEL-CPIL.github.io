"""Rewritten program using fixed-step high-order Runge-Kutta (RK45)."""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    """
    Integrates the ODE system using a fixed-step Dormand-Prince 5(4) method.
    Step size is chosen to maximize accuracy while staying within call limits.
    """
    t_start, t_end = t_span
    y = np.array(y0, dtype=float)
    
    # Calculate max steps based on call limit (2,000,000 calls)
    # RK45 uses 6 function calls per step.
    max_calls = 2000000
    max_steps = max_calls // 6
    t_total = t_end - t_start
    if t_total <= 0:
        return y
        
    # Choose step size. We want h as small as possible to minimize rel error
    # but not so small that we exceed call limit.
    # h_min = t_total / max_steps
    h_min = t_total / max_steps
    
    # Use a step size slightly larger than h_min to be safe, or just h_min.
    # However, if h is too small, floating point errors might accumulate?
    # For t=5, h=1.5e-5 is fine.
    h = h_min * 1.05  # 5% margin
    
    n_steps = int(np.ceil(t_total / h))
    h = t_total / n_steps
    
    # Dormand-Prince 5(4) coefficients
    a = np.array([
        [0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0]
    ])
    b = np.array([1/5, 9/40, 6/15, 3/4, -9/40, 1]) # b for 4th order? No, these are c.
    # Correct Dormand-Prince coefficients
    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1])
    A = np.array([
        [0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0]
    ])
    # Weights for 5th order
    w = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
    
    t = t_start
    for i in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + c[1]*h, y + h*A[1,0]*k1)
        k3 = f(t + c[2]*h, y + h*(A[2,0]*k1 + A[2,1]*k2))
        k4 = f(t + c[3]*h, y + h*(A[3,0]*k1 + A[3,1]*k2 + A[3,2]*k3))
        k5 = f(t + c[4]*h, y + h*(A[4,0]*k1 + A[4,1]*k2 + A[4,2]*k3 + A[4,3]*k4))
        k6 = f(t + c[5]*h, y + h*(A[5,0]*k1 + A[5,1]*k2 + A[5,2]*k3 + A[5,3]*k4 + A[5,4]*k5))
        
        y = y + h * (w[0]*k1 + w[1]*k2 + w[2]*k3 + w[3]*k4 + w[4]*k5 + w[5]*k6)
        t += h
        
    return y
# EVOLVE-BLOCK-END
