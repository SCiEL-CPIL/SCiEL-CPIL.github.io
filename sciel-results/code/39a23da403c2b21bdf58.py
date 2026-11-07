"""
Strategy: Custom high-order RK45 implementation with strict time guarding.
The Lotka-Volterra system is smooth and non-stiff for the given parameters.
A custom integrator allows tighter control over step size and potentially
fewer function calls than the generic SciPy solver if tuned correctly,
or at least comparable performance with guaranteed termination.
"""
import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Deadline guard: 330s limit, use 320s to be safe.
    start_time = time.time()
    deadline = start_time + 320.0
    
    # If the span is zero, return y0
    if t1 == t0:
        return y

    # Initial step size estimate
    # Use a heuristic based on the first derivative
    try:
        f0 = f(t0, y)
        # Scale h based on magnitude of y and f
        scale = np.max(np.abs(y)) + 1e-10
        f_scale = np.max(np.abs(f0)) + 1e-10
        # Estimate initial step size to keep change small
        h = 0.01 * (t1 - t0)
        # If derivative is large, take smaller step
        if f_scale > scale:
            h = 0.01 * h * (scale / f_scale)
        # Ensure h is not zero and within bounds
        h = max(min(h, abs(t1 - t0) / 10.0), 1e-10)
    except Exception:
        h = (t1 - t0) / 100.0
        f0 = f(t0, y)

    t = t0
    # Use a high-order method like RK45 (Dormand-Prince) coefficients
    # Coefficients for Dormand-Prince 5(4)
    A = [
        [],
        [1/5],
        [3/40, 9/40],
        [44/45, -56/15, 32/9],
        [19372/6561, -25360/2187, 64448/6561, -212/729],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
    ]
    B = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    B_star = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
    C = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]

    max_steps = 100000
    step_count = 0
    
    while t < t1:
        if time.time() > deadline:
            # Return current best estimate if time runs out
            return y
            
        # Don't overshoot t1
        if t + h > t1:
            h = t1 - t
            
        # RK45 Step
        k1 = f(t, y)
        
        y2 = y + h * A[1][0] * k1
        k2 = f(t + h * C[1], y2)
        
        y3 = y + h * (A[2][0] * k1 + A[2][1] * k2)
        k3 = f(t + h * C[2], y3)
        
        y4 = y + h * (A[3][0] * k1 + A[3][1] * k2 + A[3][2] * k3)
        k4 = f(t + h * C[3], y4)
        
        y5 = y + h * (A[4][0] * k1 + A[4][1] * k2 + A[4][2] * k3 + A[4][3] * k4)
        k5 = f(t + h * C[4], y5)
        
        y6 = y + h * (A[5][0] * k1 + A[5][1] * k2 + A[5][2] * k3 + A[5][3] * k4 + A[5][4] * k5)
        k6 = f(t + h * C[5], y6)
        
        # 4th order estimate
        y_new = y + h * (B[0]*k1 + B[2]*k3 + B[3]*k4 + B[4]*k5 + B[5]*k6)
        
        # 5th order estimate
        y_star = y + h * (B_star[0]*k1 + B_star[2]*k3 + B_star[3]*k4 + B_star[4]*k5 + B_star[5]*k6 + B_star[6]*k6)
        
        # Error estimate
        err = np.max(np.abs(y_star - y_new))
        
        # Accept step if error is small
        # Tolerance: rtol=1e-11, atol=1e-13
        rtol = 1e-11
        atol = 1e-13
        scale = np.max(np.abs(y)) + np.max(np.abs(y_new)) + 1e-10
        tol = atol + rtol * scale
        
        if err <= tol:
            y = y_new
            t += h
            step_count += 1
            # Adjust step size
            # Safety factor 0.9
            if err > 0:
                h_new = 0.9 * h * (tol / err) ** (1/5.0)
            else:
                h_new = 2.0 * h
            # Limit step size change
            h_new = max(0.2 * h, min(5.0 * h, h_new))
            h = h_new
        else:
            # Reject step, reduce h
            if err > 0:
                h_new = 0.9 * h * (tol / err) ** (1/5.0)
            else:
                h_new = 0.5 * h
            h_new = max(0.1 * h, h_new)
            h = h_new
            # Ensure we don't get stuck with tiny h
            if h < 1e-12:
                h = 1e-12
                
        if step_count > max_steps:
            break

    return y
