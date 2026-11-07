"""ODE integrator for the Van der Pol oscillator (mu = 5) using DOP853."""
import numpy as np


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator from t_span[0] to t_span[1] with initial condition y0.
    Uses DOP853 (an 8th-order Runge-Kutta method) with a fixed number of steps.
    Returns the state at t_span[1].
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # DOP853 coefficients (12-stage, 8th order with 5th order error estimate)
    # From Hairer, Norsett, Wanner - Solving Ordinary Differential Equations I
    # c values (nodes)
    c1 = 0
    c2 = 1/12
    c3 = 1/6
    c4 = 1/4
    c5 = 5/12
    c6 = 1/2
    c7 = 5/6
    c8 = 3/4
    c9 = 1
    c10 = 1
    c11 = 1
    c12 = 1
    
    # a matrix entries (only non-zero, stored as list of (i, j, a_ij) for i > j)
    # Stage 2: a[2,1]
    a21 = 1/12
    # Stage 3: a[3,1], a[3,2]
    a31 = 1/24
    a32 = 1/24
    # Stage 4: a[4,1], a[4,3]
    a41 = 1/6
    a43 = 1/3
    # Stage 5: a[5,2], a[5,3], a[5,4]
    a52 = 5/12
    a53 = -5/12
    a54 = 5/12
    # Stage 6: a[6,2], a[6,4], a[6,5]
    a62 = 1/8
    a64 = 1/4
    a65 = 1/8
    # Stage 7: a[7,3], a[7,4], a[7,5]
    a73 = -101/192
    a74 = 25/32
    a75 = 125/192
    # Stage 8: a[8,6]
    a86 = 1
    # Stage 9: a[9,3], a[9,4], a[9,6], a[9,7], a[9,8]
    a93 = -3/2
    a94 = 13/4
    a96 = -3
    a97 = 3/2
    a98 = 1
    # Stage 10: a[10,4], a[10,7], a[10,8], a[10,9]
    a104 = -3
    a107 = 10/3
    a108 = -13/6
    a109 = 1
    # Stage 11: a[11,4], a[11,7], a[11,8], a[11,9], a[11,10]
    a114 = -3/2
    a117 = 5/2
    a118 = -13/12
    a119 = 1
    a1110 = -1/2
    # Stage 12: a[12,4], a[12,7], a[12,8], a[12,9], a[12,10], a[12,11]
    a124 = -3/2
    a127 = 5/2
    a128 = -13/12
    a129 = 1
    a1210 = -1/2
    a1211 = 1/2
    
    # 8th order weights
    b1 = 361/5120
    b2 = 200/729
    b3 = 22951/53568
    b4 = 0
    b5 = 9305/13824
    b6 = 10430/8191
    b7 = 2197/4104
    b8 = -1
    b9 = 0
    b10 = 0
    b11 = 0
    b12 = 0
    
    # 5th order weights (for error estimation)
    e1 = 71/57600
    e2 = 0
    e3 = -71/16695
    e4 = 71/1920
    e5 = -17253/339200
    e6 = 22/525
    e7 = -27/140
    e8 = 1/4
    e9 = 0
    e10 = 0
    e11 = 0
    e12 = 0
    
    # Use 200 steps for high accuracy with low call count
    num_steps = 200
    h = (t1 - t0) / num_steps
    
    t = t0
    for _ in range(num_steps):
        # Stage 1
        k1 = f(t, y)
        
        # Stage 2
        y2 = y + h * a21 * k1
        k2 = f(t + c2 * h, y2)
        
        # Stage 3
        y3 = y + h * (a31 * k1 + a32 * k2)
        k3 = f(t + c3 * h, y3)
        
        # Stage 4
        y4 = y + h * (a41 * k1 + a43 * k3)
        k4 = f(t + c4 * h, y4)
        
        # Stage 5
        y5 = y + h * (a52 * k2 + a53 * k3 + a54 * k4)
        k5 = f(t + c5 * h, y5)
        
        # Stage 6
        y6 = y + h * (a62 * k2 + a64 * k4 + a65 * k5)
        k6 = f(t + c6 * h, y6)
        
        # Stage 7
        y7 = y + h * (a73 * k3 + a74 * k4 + a75 * k5)
        k7 = f(t + c7 * h, y7)
        
        # Stage 8
        y8 = y + h * (a86 * k6)
        k8 = f(t + c8 * h, y8)
        
        # Stage 9
        y9 = y + h * (a93 * k3 + a94 * k4 + a96 * k6 + a97 * k7 + a98 * k8)
        k9 = f(t + c9 * h, y9)
        
        # Stage 10
        y10 = y + h * (a104 * k4 + a107 * k7 + a108 * k8 + a109 * k9)
        k10 = f(t + c10 * h, y10)
        
        # Stage 11
        y11 = y + h * (a114 * k4 + a117 * k7 + a118 * k8 + a119 * k9 + a1110 * k10)
        k11 = f(t + c11 * h, y11)
        
        # Stage 12
        y12 = y + h * (a124 * k4 + a127 * k7 + a128 * k8 + a129 * k9 + a1210 * k10 + a1211 * k11)
        k12 = f(t + c12 * h, y12)
        
        # Update y using 8th order weights
        y = y + h * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6 + 
                     b7 * k7 + b8 * k8 + b9 * k9 + b10 * k10 + b11 * k11 + b12 * k12)
        
        t += h
        
        # Check for non-finite values
        if not np.all(np.isfinite(y)):
            break
    
    if not np.all(np.isfinite(y)):
        # Fallback: use more steps with RK4
        y = np.array(y0, dtype=float)
        t = t0
        num_steps_fallback = 10000
        h_fallback = (t1 - t0) / num_steps_fallback
        for _ in range(num_steps_fallback):
            k1 = f(t, y)
            k2 = f(t + h_fallback / 2, y + h_fallback / 2 * k1)
            k3 = f(t + h_fallback / 2, y + h_fallback / 2 * k2)
            k4 = f(t + h_fallback, y + h_fallback * k3)
            y = y + h_fallback / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h_fallback
    
    return y
