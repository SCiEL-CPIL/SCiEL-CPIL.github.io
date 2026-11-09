# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float, copy=True)
    
    # Fixed-step Dormand-Prince 5(4) method, using the 5th order solution.
    # This is a well-known high-order explicit RK method.
    
    # Coefficients for DP5 (Dormand-Prince 5th order)
    # c = [0, 1/5, 3/10, 4/5, 8/9, 1]
    # A matrix
    A = np.array([
        [0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0]
    ])
    
    # 5th order weights
    b5 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
    
    # Choose step size
    # For Van der Pol with mu=5, the dynamics are moderately stiff but not extremely.
    # DP5 has global error ~ C * h^5.
    # We want rel <= 1e-14.
    # Let's start with h = 0.001.
    # N = 10 / 0.001 = 10000 steps.
    # Calls per step = 6 (k1..k6).
    # Total calls = 60000.
    # log10(60000) ~ 4.78. Penalty ~ -2.39.
    # Error: (0.001)^5 * C. If C ~ 1, error ~ 1e-15. This should be sufficient.
    
    h = 0.001
    N = int(np.ceil((t1 - t0) / h))
    h = (t1 - t0) / N  # Adjust h to exactly hit t1
    
    t = t0
    
    # Precompute constants for speed
    h_c2 = h * (1/5)
    h_c3 = h * (3/10)
    h_c4 = h * (4/5)
    h_c5 = h * (8/9)
    
    for _ in range(N):
        k1 = f(t, y)
        
        y2 = y + h_c2 * k1
        k2 = f(t + 1/5, y2)
        
        y3 = y + h * (A[2,0]*k1 + A[2,1]*k2)
        k3 = f(t + 3/10, y3)
        
        y4 = y + h * (A[3,0]*k1 + A[3,1]*k2 + A[3,2]*k3)
        k4 = f(t + 4/5, y4)
        
        y5 = y + h * (A[4,0]*k1 + A[4,1]*k2 + A[4,2]*k3 + A[4,3]*k4)
        k5 = f(t + 8/9, y5)
        
        y6 = y + h * (A[5,0]*k1 + A[5,1]*k2 + A[5,2]*k3 + A[5,3]*k4 + A[5,4]*k5)
        k6 = f(t + 1, y6)
        
        y = y + h * (b5[0]*k1 + b5[2]*k3 + b5[3]*k4 + b5[4]*k5 + b5[5]*k6)
        t += h
        
    return y
# EVOLVE-BLOCK-END
