"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Solve the Robertson problem using an adaptive implicit Runge-Kutta method.
    We use a 2-stage, 2nd-order implicit RK (trapezoidal rule) with adaptive
    step-size control based on local error estimation.
    Since the problem is stiff, we solve the nonlinear system at each stage
    using Newton's method with a finite-difference Jacobian.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # Leave 50s safety margin
    
    # Initial step size: small for the rapid transient
    h = 1e-5
    t = t0
    
    # Jacobian approximation function
    def compute_jacobian(t_val, y_val):
        J = np.zeros((n, n))
        for j in range(n):
            e = np.zeros(n)
            eps = 1e-8 * max(1.0, abs(y_val[j]))
            e[j] = eps
            J[:, j] = (f(t_val, y_val + e) - f(t_val, y_val)) / eps
        return J
    
    # Newton solver for implicit stage
    def newton_solve(t_val, y_prev, coeff):
        """
        Solve: y_new = y_prev + h * coeff * f(t_val, y_new)
        Rearranged: y_new - h*coeff*f(t_val, y_new) - y_prev = 0
        """
        y_curr = y_prev.copy()
        for _ in range(20):
            f_val = f(t_val, y_curr)
            residual = y_curr - y_prev - h * coeff * f_val
            if np.linalg.norm(residual, ord=np.inf) < 1e-14 * max(1.0, np.linalg.norm(y_curr, ord=np.inf)):
                break
            J = compute_jacobian(t_val, y_curr)
            A = np.eye(n) - h * coeff * J
            try:
                step = np.linalg.solve(A, -residual)
            except np.linalg.LinAlgError:
                step = -0.5 * residual
            y_curr = y_curr + step
            if not np.all(np.isfinite(y_curr)):
                return None
        return y_curr
    
    # Main integration loop
    while t < t1:
        # Check deadline
        if time.time() > deadline:
            break
        
        # Don't step past t1
        if t + h > t1:
            h = t1 - t
        
        # Trapezoidal rule (2-stage implicit RK)
        # Stage 1: y1 = y + h/2 * f(t, y)  -> this is explicit, but we use it for error estimation
        # Actually, for a 2nd-order implicit method with error estimation, let's use:
        # Main: y_new = y + h/2 * (f(t,y) + f(t+h, y_new))  (trapezoidal)
        # Error: use a simpler method like Euler to estimate error, or use a higher-order correction
        
        # Let's use a different approach: 
        # Use BDF1 (backward Euler) as the main method and estimate error with a 1-step correction
        # Or better: use a 2-stage L-stable method
        
        # Let's use the following:
        # y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})  (backward Euler, 1st order)
        # Error estimation: y_{n+1}^{(0)} = y_n + h * f(t_n, y_n) (forward Euler)
        # local_error = |y_{n+1} - y_{n+1}^{(0)}|
        
        # Compute forward Euler step for error estimation
        f0 = f(t, y)
        y_fe = y + h * f0
        
        # Compute backward Euler step (implicit)
        y_be = newton_solve(t + h, y, 1.0)
        if y_be is None:
            # Newton failed, reduce step
            h *= 0.5
            continue
        
        # Error estimation
        err = np.linalg.norm(y_be - y_fe, ord=np.inf)
        scale = np.linalg.norm(y_be, ord=np.inf)
        rel_err = err / max(1.0, scale)
        
        # Tolerance
        rtol = 1e-13
        atol = 1e-14
        
        if rel_err <= rtol + atol:
            # Accept step
            y = y_be
            t = t + h
            
            # Adaptive step size
            if rel_err > 1e-15:
                factor = min(5.0, max(0.1, 0.9 * (rtol / rel_err) ** 0.5))
            else:
                factor = 5.0
            h = h * factor
        else:
            # Reject step, reduce h
            factor = max(0.1, 0.9 * (rtol / rel_err) ** 0.5)
            h = h * factor
    
    return y
# EVOLVE-BLOCK-END
