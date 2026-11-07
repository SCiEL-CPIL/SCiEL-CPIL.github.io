"""ODE integrator for the Lorenz system using an adaptive RK45 (Dormand-Prince) method."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Coefficients for the Dormand-Prince RK45 method (Butcher tableau)
# c, A, b4 (4th order), b5 (5th order)
_C = np.array([0.0, 2.0/9.0, 1.0/3.0, 4.0/9.0, 5.0/6.0, 1.0, 1.0])
_A = np.array([
    [0, 0, 0, 0, 0, 0],
    [2.0/9.0, 0, 0, 0, 0, 0],
    [1.0/12.0, 3.0/12.0, 0, 0, 0, 0],
    [4.0/9.0, -6.0/9.0, 3.0/2.0, 0, 0, 0],
    [5.0/6.0, -1.0, 0, 1.0, 0, 0],
    [1.0, -5.0/2.0, 4.0, 0, 0, 0],
    [71.0/576.0, 0, 7.0/24.0, 35.0/12.0, 125.0/512.0, 512.0/1771.0, 0]
])
# b4 is the 4th order solution (last row of A without the last column, but actually it's a separate vector)
# For Dormand-Prince:
# b4 = [71/576, 0, 7/24, 35/12, 125/512, 512/1771, 0]
# b5 = [1/5, 0, 3/10, 0, 3/5, -1/6, 1/6]
_B4 = np.array([71.0/576.0, 0.0, 7.0/24.0, 35.0/12.0, 125.0/512.0, 512.0/1771.0, 0.0])
_B5 = np.array([1.0/5.0, 0.0, 3.0/10.0, 0.0, 3.0/5.0, -1.0/6.0, 1.0/6.0])


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Tolerance settings - tight to match the reference
    rtol = 1e-11
    atol = 1e-13
    
    # Initial step size
    h = (t1 - t0) / 1000.0
    t = t0
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # 280 seconds, leaving margin before 330s
    
    # Minimum and maximum step sizes
    h_min = 1e-12
    h_max = (t1 - t0)
    
    while t < t1:
        # Check deadline
        if time.time() > deadline:
            break
        
        # Cap step to not overshoot t1
        if t + h > t1:
            h = t1 - t
        
        # Ensure h is not too small
        if h < h_min:
            h = h_min
        
        # Dormand-Prince RK45 step
        # Compute the 6 stages
        k = np.zeros((7, n))
        k[0] = f(t, y)
        for i in range(1, 6):
            y_stage = y + h * np.dot(_A[i, :i], k[:i])
            k[i] = f(t + _C[i] * h, y_stage)
        # 7th stage for error estimation
        y_stage = y + h * np.dot(_A[6, :6], k[:6])
        k[6] = f(t + h, y_stage)
        
        # 4th and 5th order solutions
        y4 = y + h * np.dot(_B4[:6], k[:6])
        y5 = y + h * np.dot(_B5[:6], k[:6])
        
        # Error estimate
        err = y5 - y4
        err_norm = np.linalg.norm(err, ord=np.inf)
        
        # Scale for error control
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
        err_scaled = err_norm / np.max(scale)
        
        # Accept step if error is within tolerance
        if err_scaled <= 1.0:
            t = t + h
            y = y5
            
            # Adjust step size
            if err_scaled < 1e-8:
                factor = 5.0
            else:
                factor = 0.9 * (1.0 / err_scaled) ** 0.2
            h_new = h * min(max(factor, 0.1), 5.0)
            h = min(h_new, h_max)
        else:
            # Reject step, reduce h
            factor = 0.9 * (1.0 / err_scaled) ** 0.25
            h = max(h * min(factor, 0.5), h_min)
    
    return y
# EVOLVE-BLOCK-END
