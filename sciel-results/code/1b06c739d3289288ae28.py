"""ODE integrator for the Van der Pol oscillator (mu = 5)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    total_dt = t1 - t0
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # 280s out of 300s limit
    
    # We will use a high-order explicit Runge-Kutta method (Dormand-Prince 5(4))
    # with adaptive step size control.
    # However, since we want to minimize function calls while maintaining high accuracy,
    # and the problem is somewhat stiff (mu=5), we might consider a higher order method
    # or a stiff solver. But let's first try a very accurate explicit method with small steps.
    
    # Actually, for Van der Pol with mu=5, the system is moderately stiff.
    # A fixed-step high-order method might be efficient if the step size is chosen well.
    # Let's try RK45 (Dormand-Prince) with adaptive stepping, but we need to be careful
    # about the number of function calls.
    
    # Alternative: Use a high-order fixed-step method.
    # For smooth problems, high-order methods like RK8 or RK10 can achieve high accuracy
    # with fewer steps.
    
    # Let's implement a 5th-order Runge-Kutta method (Dormand-Prince) with adaptive step size.
    # But to keep it simple and robust, let's use a fixed-step high-order method first.
    
    # Let's try a 4th-order Runge-Kutta with a very small step size, but that might be too many calls.
    # Instead, let's use a 5th-order method with adaptive step size.
    
    # Dormand-Prince coefficients (RK45)
    # a21 = 1/5
    # a31 = 3/40, a32 = 9/40
    # a41 = 44/45, a42 = -56/15, a43 = 32/9
    # a51 = 19372/6561, a52 = -25360/2187, a53 = 64448/6561, a54 = -212/729
    # a61 = 9017/3168, a62 = -355/33, a63 = 46732/5247, a64 = 49/176, a65 = -5103/18656
    # b1 = 35/384, b2 = 0, b3 = 500/1113, b4 = 125/192, b5 = -2187/6784, b6 = 11/84, b7 = 0
    # c2 = 1/5, c3 = 3/10, c4 = 4/5, c5 = 8/9, c6 = 1
    
    # For simplicity and robustness, let's use a fixed-step RK4 with a sufficiently small step.
    # But we need to balance accuracy and number of calls.
    
    # Let's estimate the required step size.
    # The Van der Pol oscillator with mu=5 has a limit cycle with period ~ 2*pi*mu^(1/3) ~ 2*pi*1.71 ~ 10.7.
    # So in [0, 10], we have less than one full period.
    # The solution is smooth, so a high-order method should work well.
    
    # Let's use a fixed-step RK4 with step size h = 1e-3. That's 10,000 steps, 40,000 function calls.
    # The score penalty for 40,000 calls is -0.5*log10(40000) ~ -0.5*4.6 = -2.3.
    # If we can get rel error ~ 1e-14, the score is -log10(1e-14) - 2.3 = 14 - 2.3 = 11.7.
    # But we can do better with fewer calls.
    
    # Let's try h = 1e-2. That's 1,000 steps, 4,000 function calls.
    # Penalty: -0.5*log10(4000) ~ -0.5*3.6 = -1.8.
    # If rel error is ~ 1e-10, score is 10 - 1.8 = 8.2.
    
    # Let's try h = 1e-1. That's 100 steps, 400 function calls.
    # Penalty: -0.5*log10(400) ~ -0.5*2.6 = -1.3.
    # If rel error is ~ 1e-6, score is 6 - 1.3 = 4.7.
    
    # So we need to balance. Let's use an adaptive step size method to minimize calls.
    
    # Implement RK45 (Dormand-Prince) with adaptive step size.
    
    # Coefficients
    a21 = 1/5
    a31 = 3/40; a32 = 9/40
    a41 = 44/45; a42 = -56/15; a43 = 32/9
    a51 = 19372/6561; a52 = -25360/2187; a53 = 64448/6561; a54 = -212/729
    a61 = 9017/3168; a62 = -355/33; a63 = 46732/5247; a64 = 49/176; a65 = -5103/18656
    b1 = 35/384; b2 = 0; b3 = 500/1113; b4 = 125/192; b5 = -2187/6784; b6 = 11/84
    c2 = 1/5; c3 = 3/10; c4 = 4/5; c5 = 8/9; c6 = 1
    
    # Error coefficients (b - b_error)
    e1 = 71/57600; e2 = 0; e3 = -71/16695; e4 = 71/1920; e5 = -17253/339200; e6 = 22/525; e7 = -1/40
    
    rtol = 1e-10
    atol = 1e-12
    
    t = t0
    h = 0.01  # Initial step size
    max_h = 0.1
    min_h = 1e-6
    
    while t < t1:
        if time.time() > deadline:
            break
        
        # Clamp step size
        h = min(h, t1 - t)
        h = max(h, min_h)
        
        # Compute stages
        k1 = f(t, y)
        k2 = f(t + c2*h, y + h*(a21*k1))
        k3 = f(t + c3*h, y + h*(a31*k1 + a32*k2))
        k4 = f(t + c4*h, y + h*(a41*k1 + a42*k2 + a43*k3))
        k5 = f(t + c5*h, y + h*(a51*k1 + a52*k2 + a53*k3 + a54*k4))
        k6 = f(t + c6*h, y + h*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))
        
        # 5th order solution
        y_new = y + h*(b1*k1 + b3*k3 + b4*k4 + b5*k5 + b6*k6)
        
        # Error estimate
        err = h*(e1*k1 + e3*k3 + e4*k4 + e5*k5 + e6*k6 + e7*k6)
        
        # Check if all finite
        if not np.all(np.isfinite(y_new)) or not np.all(np.isfinite(err)):
            h *= 0.5
            continue
        
        # Error norm
        err_norm = np.linalg.norm(err / (atol + rtol * np.maximum(np.abs(y), np.abs(y_new))))
        
        if err_norm <= 1.0:
            # Accept step
            t += h
            y = y_new
            
            # Adjust step size
            if err_norm > 0:
                h_new = h * min(5.0, max(0.1, 0.9 * err_norm**(-0.2)))
            else:
                h_new = h * 5.0
            h = min(h_new, max_h)
        else:
            # Reject step
            h = h * max(0.1, 0.9 * err_norm**(-0.25))
            h = max(h, min_h)
    
    return y
# EVOLVE-BLOCK-END
