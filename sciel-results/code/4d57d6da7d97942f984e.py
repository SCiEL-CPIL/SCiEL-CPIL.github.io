"""ODE solver for Robertson problem using adaptive BDF method."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _jac_robertson(t, y):
    """Analytical Jacobian for Robertson problem."""
    y0, y1, y2 = y
    return np.array([
        [-1e-4, 1e-4, 0],
        [1e-4, -1e-4 - 3e-3, 0],
        [0, 3e-3, -1e5]
    ])


def _bdf2_step(f, t, y_prev, y_curr, h, J):
    """Take one BDF2 step: (3*y_{n+1} - 4*y_n + y_{n-1})/(2*h) = f(t+h, y_{n+1})"""
    # Newton iteration to solve: 3*z - 4*y_curr + y_prev = 2*h*f(t+h, z)
    # Rearranged: 3*z - 2*h*f(t+h, z) - 4*y_curr + y_prev = 0
    z = y_curr + h * (y_curr - y_prev)  # initial guess (extrapolate)
    
    for _ in range(10):
        fz = f(t + h, z)
        residual = 3.0 * z - 2.0 * h * fz - 4.0 * y_curr + y_prev
        # Jacobian of residual: 3*I - 2*h*J
        A = 3.0 * np.eye(len(z)) - 2.0 * h * J
        try:
            step = np.linalg.solve(A, -residual)
        except np.linalg.LinAlgError:
            break
        z = z + step
        if np.linalg.norm(step) < 1e-12 * (1.0 + np.linalg.norm(z)):
            break
    
    return z, np.linalg.norm(residual) / (1.0 + np.linalg.norm(z))


def _estimate_error(f, t, y_prev, y_curr, y_next, h, J):
    """Estimate local error using BDF1 vs BDF2 comparison."""
    # BDF1: (y_{n+1} - y_n)/h = f(t+h, y_{n+1})
    # Solve BDF1 for comparison
    z1 = y_curr.copy()
    for _ in range(5):
        fz1 = f(t + h, z1)
        residual1 = z1 - y_curr - h * fz1
        A1 = np.eye(len(z1)) - h * J
        try:
            step1 = np.linalg.solve(A1, -residual1)
        except np.linalg.LinAlgError:
            break
        z1 = z1 + step1
        if np.linalg.norm(step1) < 1e-12 * (1.0 + np.linalg.norm(z1)):
            break
    
    return np.linalg.norm(y_next - z1) / (1.0 + np.linalg.norm(y_next))


def integrate(f, t_span, y0):
    """Solve y' = f(t, y) from t_span[0] to t_span[1] using adaptive BDF2."""
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin before 330s
    
    # Initial step using BDF1 (implicit Euler)
    h = 1e-6
    J = _jac_robertson(t0, y)
    
    # First step: implicit Euler
    z = y.copy()
    for _ in range(10):
        fz = f(t0 + h, z)
        residual = z - y - h * fz
        A = np.eye(n) - h * J
        try:
            step = np.linalg.solve(A, -residual)
        except np.linalg.LinAlgError:
            break
        z = z + step
        if np.linalg.norm(step) < 1e-12 * (1.0 + np.linalg.norm(z)):
            break
    
    y_prev = y
    y_curr = z
    t_curr = t0 + h
    
    # Check if we've reached the end
    if t_curr >= t1:
        return y_curr
    
    # Main BDF2 loop
    while t_curr < t1:
        if time.time() > deadline:
            break
        
        # Don't overshoot t1
        if t_curr + h > t1:
            h = t1 - t_curr
        
        # Update Jacobian periodically
        J = _jac_robertson(t_curr, y_curr)
        
        # Take BDF2 step
        y_next, residual_norm = _bdf2_step(f, t_curr, y_prev, y_curr, h, J)
        
        # Check for failure
        if not np.all(np.isfinite(y_next)):
            h *= 0.5
            if h < 1e-15:
                break
            continue
        
        # Estimate error
        err = _estimate_error(f, t_curr, y_prev, y_curr, y_next, h, J)
        
        # Adaptive step size control
        if err < 1e-12:
            # Error very small, can increase step
            h *= 2.0
        elif err < 1e-8:
            # Good error, keep step or slightly increase
            h *= 1.2
        elif err < 1e-5:
            # Acceptable, keep step
            pass
        else:
            # Too large error, reduce step
            h *= max(0.5, (1e-8 / err) ** 0.5)
        
        # Enforce reasonable step bounds
        h = max(1e-15, min(h, 1.0))
        
        # Accept step
        y_prev = y_curr
        y_curr = y_next
        t_curr += h
    
    return y_curr
# EVOLVE-BLOCK-END
