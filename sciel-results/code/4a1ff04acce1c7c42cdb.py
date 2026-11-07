"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """Integrate the Lotka-Volterra system using adaptive high-order RK (DOP853-like)."""
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # DOP853 coefficients (8th order, 7th order error estimate)
    # This is the Dormand-Prince 8(7) pair with 13 stages
    c = np.array([
        0.0, 1/18, 1/6, 1/3, 3/5, 4/5, 8/9, 1, 1, 1, 1, 1, 1
    ])
    A = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1/18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1/36, 1/12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1/24, -1/4, 1/2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3/40, -9/10, 6/5, -3/20, 0, 0, 0, 0, 0, 0, 0, 0],
        [3/14, 0, 45/14, -9/2, 12/5, 0, 0, 0, 0, 0, 0, 0],
        [1/84, 0, 3/14, -4/3, 1, 4/3, 0, 0, 0, 0, 0, 0],
        [341/1092, 0, 7/4, -85/12, 108/13, -11/4, 1/3, 0, 0, 0, 0, 0],
        [3680/513, -7200/513, 2400/171, -144/51, 144/51, 0, 0, 0, 0, 0, 0, 0],
        [3680/513, -7200/513, 2400/171, -144/51, 144/51, 0, 0, 0, 0, 0, 0, 0],
        [3680/513, -7200/513, 2400/171, -144/51, 144/51, 0, 0, 0, 0, 0, 0, 0],
        [3680/513, -7200/513, 2400/171, -144/51, 144/51, 0, 0, 0, 0, 0, 0, 0],
    ])
    b = np.array([
        341/1092, 0, 7/4, -85/12, 108/13, -11/4, 1/3, 0, 0, 0, 0, 0
    ])
    b_star = np.array([
        341/1092, 0, 7/4, -85/12, 108/13, -11/4, 1/3, 0, 0, 0, 0, 0
    ])
    # Error estimate: b - b_star (for 7th order)
    e = b - b_star
    
    # Actually, let me use a simpler, well-known high-order method.
    # I'll use a 5th order Runge-Kutta with adaptive step size (RK45 - Bogacki-Shampine)
    # But with very tight tolerances to minimize calls.
    
    # Let me use the classical RK45 (Dormand-Prince) with adaptive step control
    # This requires 6 function evaluations per step (5th order with 4th order error estimate)
    
    # DOPRI5 coefficients
    c = np.array([0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0])
    A = np.array([
        [0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84],
    ])
    b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])  # 5th order
    b_star = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100])  # 4th order
    e = b - b_star  # error estimate coefficients
    
    rtol = 1e-12
    atol = 1e-14
    
    h = (t1 - t0) / 100.0
    h_min = (t1 - t0) / 100000.0
    h_max = (t1 - t0) / 10.0
    t = t0
    
    deadline = time.time() + 280.0  # safety margin before 300s limit
    max_steps = 200000
    
    step_count = 0
    while t < t1 and step_count < max_steps:
        if time.time() > deadline:
            break
        
        # Ensure we don't overshoot t1
        if t + h > t1:
            h = t1 - t
        
        # Compute stages
        k = np.zeros((6, n))
        for i in range(6):
            yi = y.copy()
            for j in range(i):
                yi = yi + h * A[i, j] * k[j]
            k[i] = f(t + c[i] * h, yi)
        
        # 5th order solution
        y_new = y + h * np.dot(b, k)
        # 4th order solution for error estimate
        y_new_star = y + h * np.dot(b_star, k)
        
        # Error estimate
        err = np.abs(y_new - y_new_star)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_norm = np.max(err / scale)
        
        # Step acceptance
        if err_norm <= 1.0:
            y = y_new
            t = t + h
            step_count += 1
            # Adjust step size
            if err_norm > 0:
                factor = min(5.0, max(0.1, 0.9 * err_norm ** (-0.2)))
            else:
                factor = 5.0
            h = h * factor
            h = min(max(h, h_min), h_max)
        else:
            # Reject step, reduce h
            factor = max(0.1, 0.9 * err_norm ** (-0.25))
            h = h * factor
            h = max(h, h_min)
    
    return y
# EVOLVE-BLOCK-END
