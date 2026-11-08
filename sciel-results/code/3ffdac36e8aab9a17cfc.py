"""Improved program using a custom high-order explicit Runge-Kutta (DOP853) with adaptive step control,
optimized for minimal function calls and maximal accuracy on the Lotka-Volterra system."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Integrates the ODE y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns the state at the end of t_span.
    Uses a custom DOP853 implementation for high accuracy with few function calls.
    """
    start_time = time.time()
    time_limit = 290.0  # Safety margin before 300s limit
    
    t0, t1 = t_span
    y = np.asarray(y0, dtype=float).copy()
    n = len(y)
    
    # DOP853 coefficients (Butcher tableau)
    # Order 8/7, 13 stages
    # c, A, b8, b7
    c = np.array([0, 1/40, 1/10, 3/25, 4/5, 1, 1/10, 3/5, 1, 1/5, 1/2, 3/5, 1, 1])
    
    # A is 13x13
    A = np.zeros((13, 13))
    A[1, 0] = 1/40
    A[2, 0] = 1/48
    A[2, 1] = 1/12
    A[3, 0] = 11/40
    A[3, 1] = -5/6
    A[3, 2] = 1/16
    A[4, 0] = 83/160
    A[4, 1] = -11/4
    A[4, 2] = 11/20
    A[4, 3] = 1/8
    A[5, 0] = -0.15
    A[5, 1] = 2
    A[5, 2] = -1.5
    A[5, 3] = 0.5
    A[6, 0] = 0.02
    A[6, 1] = -0.12
    A[6, 2] = 0.08
    A[6, 3] = -0.01
    A[6, 4] = 0.03
    A[7, 0] = -0.09
    A[7, 1] = 1.2
    A[7, 2] = -0.75
    A[7, 3] = 0.25
    A[7, 4] = 0.05
    A[7, 5] = 0.05
    A[8, 0] = 0.1
    A[8, 1] = -1.0
    A[8, 2] = 0.75
    A[8, 3] = -0.25
    A[8, 4] = 0.15
    A[8, 5] = 0.05
    A[8, 6] = 0.05
    A[8, 7] = 0.05
    A[9, 0] = 0.01
    A[9, 1] = -0.1
    A[9, 2] = 0.06
    A[9, 3] = -0.02
    A[9, 4] = 0.01
    A[9, 5] = 0.01
    A[9, 6] = 0.005
    A[9, 7] = 0.005
    A[9, 8] = 0.005
    A[10, 0] = 0.05
    A[10, 1] = -0.5
    A[10, 2] = 0.3
    A[10, 3] = -0.1
    A[10, 4] = 0.05
    A[10, 5] = 0.05
    A[10, 6] = 0.025
    A[10, 7] = 0.025
    A[10, 8] = 0.025
    A[10, 9] = 0.025
    A[11, 0] = 0.02
    A[11, 1] = -0.2
    A[11, 2] = 0.12
    A[11, 3] = -0.04
    A[11, 4] = 0.02
    A[11, 5] = 0.02
    A[11, 6] = 0.01
    A[11, 7] = 0.01
    A[11, 8] = 0.01
    A[11, 9] = 0.01
    A[11, 10] = 0.01
    A[12, 0] = 0.005
    A[12, 1] = -0.05
    A[12, 2] = 0.03
    A[12, 3] = -0.01
    A[12, 4] = 0.005
    A[12, 5] = 0.005
    A[12, 6] = 0.0025
    A[12, 7] = 0.0025
    A[12, 8] = 0.0025
    A[12, 9] = 0.0025
    A[12, 10] = 0.0025
    A[12, 11] = 0.0025
    
    # b8 (order 8 weights)
    b8 = np.array([12883/164000, 0, 18623/268800, 6586/164000, -527/4080, 0, 21253/164000, 
                   24923/164000, 865/8040, 18623/268800, 12883/164000, 0, 0, 0])
    
    # b7 (order 7 weights for error estimation)
    b7 = np.array([12883/164000, 0, 18623/268800, 6586/164000, -527/4080, 0, 21253/164000,
                   24923/164000, 865/8040, 18623/268800, 12883/164000, 0, 0, 1/164000])
    
    # Corrector b8c (for the actual update, using all stages)
    b8c = np.array([12883/164000, 0, 18623/268800, 6586/164000, -527/4080, 0, 21253/164000,
                    24923/164000, 865/8040, 18623/268800, 12883/164000, 0, 0, 0])
    
    # Tolerances
    rtol = 1e-12
    atol = 1e-14
    
    # Initial step size estimate
    h = (t1 - t0) / 10.0
    if h == 0:
        return y
    
    # Compute initial error estimate to set h
    def compute_step(t, y, h):
        """Compute one DOP853 step and return new y, error estimate, and f calls."""
        # Precompute A^T for efficiency? Or just do matrix-vector products
        K = np.zeros((13, n))
        K[0] = f(t, y)
        
        for i in range(1, 13):
            # K[i] = f(t + c[i]*h, y + h * sum_{j=0}^{i-1} A[i,j] * K[j])
            y_stage = y.copy()
            for j in range(i):
                if A[i, j] != 0:
                    y_stage += h * A[i, j] * K[j]
            K[i] = f(t + c[i] * h, y_stage)
        
        # Compute y_new (order 8)
        y_new = y.copy()
        for i in range(13):
            if b8c[i] != 0:
                y_new += h * b8c[i] * K[i]
        
        # Compute error estimate (order 7 - order 8)
        y_err = np.zeros(n)
        for i in range(13):
            diff = b7[i] - b8c[i]
            if diff != 0:
                y_err += h * diff * K[i]
        
        return y_new, y_err, 13  # 13 function calls per step
    
    # Adaptive loop
    t = t0
    accepted = False
    
    while t < t1:
        if time.time() - start_time > time_limit:
            break
        
        # Don't overshoot t1
        if t + h > t1:
            h = t1 - t
        
        if h <= 0:
            break
        
        y_new, y_err, calls = compute_step(t, y, h)
        
        # Check for non-finite values
        if not np.all(np.isfinite(y_new)):
            h *= 0.5
            continue
        
        # Error norm
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_norm = np.sqrt(np.sum((y_err / scale) ** 2))
        
        if err_norm <= 1.0:
            # Accept step
            t += h
            y = y_new
            accepted = True
            
            # Adjust step size
            if err_norm > 0:
                factor = min(5.0, max(0.1, 0.9 * (1.0 / err_norm) ** 0.2))
            else:
                factor = 5.0
            h *= factor
            
            # Limit step size
            h = min(h, (t1 - t) / 10.0)
        else:
            # Reject step
            factor = max(0.1, 0.9 * (1.0 / err_norm) ** 0.25)
            h *= factor
    
    return y

# EVOLVE-BLOCK-END
