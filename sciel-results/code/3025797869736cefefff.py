"""Adaptive Dormand-Prince 5(4) integrator for Lotka-Volterra.
Strategy: Implement a custom adaptive DP5 integrator with very tight tolerances
(rtol=1e-13, atol=1e-15) to achieve high accuracy. This structural change replaces
the fragile fixed-step DOP853 with a robust adaptive method that adjusts step size
based on local error, ensuring efficiency and precision.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Dormand-Prince 5(4) coefficients
# c, a, b (5th order), b_star (4th order)
# Reference: Dormand and Prince, 1980
_c = np.array([
    0.0,
    1.0/5.0,
    3.0/10.0,
    4.0/5.0,
    8.0/9.0,
    1.0,
    1.0
])

# A matrix (7x7, but only upper triangular part is non-zero for explicit RK)
# We store the full matrix for simplicity, or just the coefficients a_ij
# k_i = f(t + c_i h, y + h * sum_{j < i} a_ij k_j)
_a = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [1/5, 0, 0, 0, 0, 0, 0],
    [3/40, 9/40, 0, 0, 0, 0, 0],
    [44/45, -56/15, 32/9, 0, 0, 0, 0],
    [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
    [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
])

# b coefficients for 5th order solution
_b = np.array([
    35/384.0,
    0.0,
    500/1113.0,
    125/192.0,
    -2187/6784.0,
    11/84.0,
    0.0
])

# b_star coefficients for 4th order solution (for error estimation)
_b_star = np.array([
    5179/57600.0,
    0.0,
    7571/16695.0,
    393/640.0,
    -92097/339200.0,
    187/2100.0,
    1/40.0
])

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = y0.shape[0]
    T = t1 - t0
    
    # Wall-clock guard
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Tolerances
    rtol = 1e-13
    atol = 1e-15
    
    # Initial step size estimate
    # Heuristic: try a small step, then adjust
    h = T / 1000.0
    if h == 0:
        h = 1e-6
    t = t0
    y = y0.copy()
    
    # Safety factors for step size adjustment
    safety = 0.9
    min_factor = 0.2
    max_factor = 10.0
    
    # Maximum number of steps to prevent infinite loops
    max_steps = 100000
    
    steps = 0
    while t < t1 and steps < max_steps:
        if time.time() > deadline:
            break
            
        # Ensure we don't step past t1
        if t + h > t1:
            h = t1 - t
            if h <= 0:
                break
                
        # Perform DP5 step
        k = np.zeros((7, n))
        
        # Stage 1
        k[0] = f(t, y)
        
        # Stages 2-7
        for i in range(1, 7):
            y_stage = y + h * np.dot(_a[i, :i], k[:i])
            k[i] = f(t + _c[i] * h, y_stage)
            
        # 5th order solution
        y_new = y + h * np.dot(_b, k)
        
        # 4th order solution (error estimate)
        y_err = y + h * np.dot(_b_star, k)
        
        # Error vector
        e = y_new - y_err
        
        # Error norm
        # Scale by atol + rtol * max(|y|, |y_new|)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_norm = np.linalg.norm(e / scale) / np.sqrt(n)
        
        # Step size adjustment
        if err_norm <= 1.0:
            # Accept step
            y = y_new
            t += h
            steps += 1
            
            # Adjust h for next step
            if err_norm > 0:
                factor = safety * (1.0 / err_norm) ** 0.2
            else:
                factor = max_factor
            factor = min(max(factor, min_factor), max_factor)
            h *= factor
        else:
            # Reject step, reduce h
            factor = safety * (1.0 / err_norm) ** 0.25
            factor = max(factor, min_factor)
            h *= factor
            
            # If h becomes too small, break to avoid infinite loop
            if h < 1e-15:
                break

    return y
# EVOLVE-BLOCK-END
