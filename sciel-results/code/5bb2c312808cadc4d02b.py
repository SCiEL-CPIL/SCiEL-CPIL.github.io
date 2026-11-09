# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate the Lorenz system using a high-order Runge-Kutta method with
    adaptive step size control.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 290
    
    # Use DOP853 coefficients for 8th order accuracy with 5th order error estimate
    # This is the standard DOP853 method
    
    # DOP853 coefficients
    c = [0.0, 1/35, 2/35, 3/35, 1/10, 1/6, 2/35, 1/7, 1/3, 12/17, 1.0, 1.0, 1.0]
    
    # Stage matrix A
    A = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1/35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2/35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3/35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1/10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1/6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2/35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1/7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1/3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [12/17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    
    # Actually, the DOP853 method has a more complex A matrix. Let me use a simpler
    # but still high-order approach.
    
    # Use a fixed-step 8th order Runge-Kutta method with many small steps
    # This is simpler and more reliable
    
    # For the Lorenz system on [0, 2], we need good accuracy
    # Let's use a high number of steps with a 4th order method, but with
    # step size chosen to give high accuracy
    
    # Actually, let me use a proper adaptive high-order method
    
    # Dormand-Prince 5(4) method coefficients
    c_dp = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    A_dp = [
        [0, 0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    ]
    b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    b4 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
    
    rtol = 1e-11
    atol = 1e-13
    
    h = 0.01  # Initial step size
    t = t0
    call_count = 0
    max_calls = 1500000
    
    while t < t1 - 1e-14:
        if call_count >= max_calls:
            break
        if time.time() > deadline:
            break
        
        h = min(h, t1 - t)
        
        # Compute RK stages
        k1 = f(t, y)
        call_count += 1
        k2 = f(t + c_dp[1]*h, y + h*A_dp[1, 0]*k1)
        call_count += 1
        k3 = f(t + c_dp[2]*h, y + h*(A_dp[2, 0]*k1 + A_dp[2, 1]*k2))
        call_count += 1
        k4 = f(t + c_dp[3]*h, y + h*(A_dp[3, 0]*k1 + A_dp[3, 1]*k2 + A_dp[3, 2]*k3))
        call_count += 1
        k5 = f(t + c_dp[4]*h, y + h*(A_dp[4, 0]*k1 + A_dp[4, 1]*k2 + A_dp[4, 2]*k3 + A_dp[4, 3]*k4))
        call_count += 1
        k6 = f(t + c_dp[5]*h, y + h*(A_dp[5, 0]*k1 + A_dp[5, 1]*k2 + A_dp[5, 2]*k3 + A_dp[5, 3]*k4 + A_dp[5, 4]*k5))
        call_count += 1
        k7 = f(t + c_dp[6]*h, y + h*(A_dp[6, 0]*k1 + A_dp[6, 2]*k3 + A_dp[6, 3]*k4 + A_dp[6, 4]*k5 + A_dp[6, 5]*k6))
        call_count += 1
        
        # 5th order solution (use k1-k6, not k7)
        y5 = y + h*(b5[0]*k1 + b5[2]*k3 + b5[3]*k4 + b5[4]*k5 + b5[5]*k6)
        # 4th order solution
        y4 = y + h*(b4[0]*k1 + b4[2]*k3 + b4[3]*k4 + b4[4]*k5 + b4[5]*k6 + b4[6]*k7)
        
        # Error estimate
        err = np.max(np.abs(y5 - y4) / (atol + rtol * np.max(np.abs(y5))))
        
        if err <= 1.0:
            y = y5
            t += h
            # Adjust step size
            if err > 1e-8:
                h *= 0.9 * (1.0/err)**0.2
            else:
                h *= 1.3
            h = min(h, (t1 - t) / 2)
        else:
            h *= 0.9 * (1.0/err)**0.25
            h = max(h, 1e-12)
    
    return y
# EVOLVE-BLOCK-END
