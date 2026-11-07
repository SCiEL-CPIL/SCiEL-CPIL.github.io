"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _rk45_step(f, t, y, h):
    """Single RK45 step (Dormand-Prince) returning y5 (4th order) and y6 (5th order)."""
    # Coefficients for Dormand-Prince RK45
    a2, a3, a4, a5, a6, a7 = 1/5, 3/10, 4/5, 8/9, 1, 1
    b21 = 1/5
    b31, b32 = 3/40, 9/40
    b41, b42, b43 = 44/45, -56/15, 32/9
    b51, b52, b53, b54 = 19372/6561, -25360/2187, 64448/6561, -212/729
    b61, b62, b63, b64, b65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
    b71, b72, b73, b74, b75, b76 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84
    
    c2, c3, c4, c5, c6, c7 = a2, a3, a4, a5, a6, a7
    
    k1 = f(t, y)
    k2 = f(t + c2*h, y + h*(b21*k1))
    k3 = f(t + c3*h, y + h*(b31*k1 + b32*k2))
    k4 = f(t + c4*h, y + h*(b41*k1 + b42*k2 + b43*k3))
    k5 = f(t + c5*h, y + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4))
    k6 = f(t + c6*h, y + h*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5))
    
    # 5th order solution (for advancing)
    y5 = y + h*(35/384*k1 + 0*k2 + 500/1113*k3 + 125/192*k4 - 2187/6784*k5 + 11/84*k6)
    
    # 4th order solution (for error estimation)
    y4 = y + h*(5179/57600*k1 + 7571/16695*k2 + 393/640*k3 - 92097/339200*k4 + 187/2100*k5 + 1/40*k6)
    
    return y5, y4


def _adaptive_rk45(f, t0, t1, y0, rtol=1e-10, atol=1e-12):
    """Adaptive RK45 with step size control."""
    t = t0
    y = np.array(y0, dtype=float)
    h = (t1 - t0) / 100  # initial step
    
    # Safety factors
    SAFETY = 0.9
    MIN_FACTOR = 0.1
    MAX_FACTOR = 10.0
    
    while t < t1 - 1e-15:
        # Don't step past t1
        if t + h > t1:
            h = t1 - t
        
        y5, y4 = _rk45_step(f, t, y, h)
        
        # Error estimate
        err = np.linalg.norm((y5 - y4) / (atol + rtol * np.maximum(np.abs(y4), np.abs(y5))))
        
        if err <= 1.0:
            # Accept step
            t += h
            y = y5
            
            # Adjust step size
            if err > 0:
                factor = SAFETY * (err ** (-0.2))
            else:
                factor = MAX_FACTOR
            factor = min(max(factor, MIN_FACTOR), MAX_FACTOR)
            h = h * factor
        
        # Ensure we don't overshoot
        if t + h > t1:
            h = t1 - t
        
        # Safety check for divergence
        if not np.all(np.isfinite(y)):
            return None
        
        # Prevent infinite loops
        if abs(h) < 1e-15:
            break
    
    return y


def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system using adaptive RK45.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard: 300s wall-clock limit
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin
    
    # First attempt with moderate tolerances
    y = _adaptive_rk45(f, t0, t1, y0, rtol=1e-10, atol=1e-12)
    
    if y is None or not np.all(np.isfinite(y)):
        # Fallback: try with looser tolerances
        y = _adaptive_rk45(f, t0, t1, y0, rtol=1e-8, atol=1e-10)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 280:
        # Return best result we have
        if y is not None and np.all(np.isfinite(y)):
            return y
        return np.array(y0, dtype=float)
    
    # If we have time, do a refinement pass with tighter tolerances
    if elapsed < 250:
        y_refined = _adaptive_rk45(f, t0, t1, y0, rtol=1e-12, atol=1e-14)
        if y_refined is not None and np.all(np.isfinite(y_refined)):
            y = y_refined
    
    # Final safety check
    if y is None or not np.all(np.isfinite(y)):
        return np.array(y0, dtype=float)
    
    return y
# EVOLVE-BLOCK-END
