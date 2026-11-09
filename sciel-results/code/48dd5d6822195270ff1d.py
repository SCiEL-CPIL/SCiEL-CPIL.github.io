"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system using a high-order explicit Runge-Kutta method
    with adaptive step size control.
    
    The system is:
      y0' = 1.5 y0 - y0 y1
      y1' = -3 y1 + y0 y1
    
    We use the 5(4) Dormand-Prince method (RK45) with tight tolerances to achieve
    high accuracy with a reasonable number of function calls.
    
    Parameters:
    -----------
    f : callable
        Function f(t, y) returning y' as an array.
    t_span : tuple
        (t_start, t_end)
    y0 : array_like
        Initial condition.
        
    Returns:
    --------
    y_end : ndarray
        The state at t_end.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Tolerances for adaptive step size control
    rtol = 1e-12
    atol = 1e-14
    
    # Dormand-Prince coefficients (RK45)
    # Butcher tableau
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
    b4 = np.array([51/715, 0, 300/469, 180/239, -9/50, 2/55, 0])
    b5 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
    
    # Initial step size estimate
    h = (t1 - t0) / 1000
    if h <= 0:
        return y
    
    t = t0
    
    while t < t1:
        # Don't step past t1
        if t + h > t1:
            h = t1 - t
        if h <= 0:
            break
        
        # Compute stages
        k = np.zeros((7, n))
        k[0] = f(t, y)
        for i in range(1, 7):
            y_stage = y.copy()
            for j in range(i):
                y_stage += h * A[i, j] * k[j]
            k[i] = f(t + c[i] * h, y_stage)
        
        # 4th and 5th order solutions
        y4 = y + h * (b4 @ k)
        y5 = y + h * (b5 @ k)
        
        # Error estimate
        err = np.linalg.norm(y5 - y4)
        scale = atol + rtol * np.maximum(np.abs(y4), np.abs(y5))
        err_norm = np.linalg.norm((y5 - y4) / scale)
        
        if err_norm <= 1.0:
            # Accept step
            t += h
            y = y5  # Use 5th order for higher accuracy
            
            # Adjust step size
            if err_norm > 1e-5:
                factor = 0.9 * (err_norm ** (-0.2))
            else:
                factor = 5.0
            h *= min(max(factor, 0.2), 5.0)
        else:
            # Reject step, reduce h
            factor = 0.9 * (err_norm ** (-0.25))
            h *= max(factor, 0.1)
        
        # Safety check
        if not np.all(np.isfinite(y)):
            # Fall back to a simple fixed-step RK4 if something went wrong
            y = _fallback_rk4(f, t0, t1, y0, 50000)
            return y
    
    return y


def _fallback_rk4(f, t0, t1, y0, steps):
    """Simple fixed-step RK4 as a fallback."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y
# EVOLVE-BLOCK-END
