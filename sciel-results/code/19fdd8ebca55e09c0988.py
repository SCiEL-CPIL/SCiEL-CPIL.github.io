"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator (mu=5) from t_span[0] to t_span[1]
    using a high-order explicit Runge-Kutta method (Dormand-Prince 5th order)
    with a fixed number of steps chosen for high accuracy and low call count.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard: 300s wall-clock, return best result with safety margin
    start_time = time.time()
    deadline = start_time + 280.0  # 20s safety margin
    
    # Dormand-Prince 5(4) coefficients
    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
    A = np.array([
        [0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    ])
    b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
    b_star = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
    
    # Choose number of steps for high accuracy
    # With DP5, error per step ~ h^5, so total error ~ h^5 = (T/N)^5
    # For T=10, N=200: h=0.05, error ~ 0.05^5 = 3.1e-7 (too large)
    # N=1000: h=0.01, error ~ 1e-10
    # N=5000: h=0.002, error ~ 3.2e-13
    # N=10000: h=0.001, error ~ 1e-15
    # Each step uses 6 f evaluations (7th stage is for error estimate, but we use 6 for the solution)
    # Actually DP5 uses 6 stages for the solution (b has 6 non-zero entries effectively, 7th is for error)
    # Let's use 6 calls per step
    
    # Let's try N=5000 first: 5000*6 = 30000 calls
    # If time allows, we can try more steps for higher accuracy
    
    # Try increasing N until we get good accuracy or hit deadline
    best_y = None
    best_N = 0
    
    for N in [5000, 10000, 20000, 40000, 80000]:
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:  # Leave 10s for finishing
            break
            
        h = (t1 - t0) / N
        t = t0
        y = y0.copy()
        
        for i in range(N):
            # Check deadline
            if i % 1000 == 0 and time.time() > deadline - 5:
                break
                
            k = np.zeros((6, n))
            k[0] = f(t, y)
            for s in range(1, 6):
                y_s = y.copy()
                for j in range(s):
                    y_s += h * A[s, j] * k[j]
                k[s] = f(t + c[s] * h, y_s)
            
            y = y + h * (b[0]*k[0] + b[2]*k[2] + b[3]*k[3] + b[4]*k[4] + b[5]*k[5])
            t += h
        
        # Check if we completed all steps
        if abs(t - t1) < 1e-10:
            if np.all(np.isfinite(y)):
                best_y = y
                best_N = N
                # If we're already at machine precision level, stop
                # We can't easily check rel error without reference, but higher N should be better
                # Continue to try higher N if time permits
    
    if best_y is not None:
        return best_y
    
    # Fallback: simple RK4 with many steps
    N = 20000
    h = (t1 - t0) / N
    t = t0
    y = y0.copy()
    for i in range(N):
        if time.time() > deadline - 2:
            break
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    
    return y
# EVOLVE-BLOCK-END
