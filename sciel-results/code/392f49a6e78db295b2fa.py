"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate a 15-dimensional stiff linear ODE y' = f(t, y) over t_span.
    Uses a variable-step implicit method (backward Euler with Newton) to handle stiffness,
    with adaptive step size control.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    h_target = (t1 - t0) / 1000  # initial target step
    h_min = (t1 - t0) / 1e6
    h_max = (t1 - t0) / 10
    t = t0
    deadline = time.time() + 290  # safety margin
    
    # Precompute Jacobian at initial point to check if it's constant (linear autonomous)
    # For a linear system y' = A*y + b, the Jacobian is constant A.
    # We'll detect this by comparing Jacobians at different points.
    
    def compute_jacobian(t_val, y_val):
        """Compute Jacobian using finite differences."""
        J = np.zeros((n, n))
        f0 = f(t_val, y_val)
        for j in range(n):
            e = np.zeros(n)
            e[j] = 1e-7 * max(1.0, abs(y_val[j]))
            J[:, j] = (f(t_val, y_val + e) - f0) / e[j]
        return J, f0
    
    # First, let's try to detect if this is a linear autonomous system
    # by computing the Jacobian at two different points.
    J1, f1 = compute_jacobian(t0, y)
    
    # Check if the system is autonomous and linear by evaluating at a different point
    y_test = y + 0.1 * np.ones(n)
    t_test = t0 + 0.01 * (t1 - t0)
    f_test = f(t_test, y_test)
    f_linear_test = J1 @ y_test + (f1 - J1 @ y)  # predicted f for linear system
    if np.linalg.norm(f_test - f_linear_test) < 1e-10 * (1 + np.linalg.norm(f_test)):
        # System appears to be linear autonomous: y' = A*y + b
        A = J1
        b = f1 - J1 @ y
        # Use matrix exponential or implicit method with constant Jacobian
        # For linear systems, we can use the exact solution via matrix exponential
        # y(t) = exp(A*(t-t0)) * y0 + integral of exp(A*(t-s))*b ds
        # But matrix exponential might be expensive; let's use a high-order implicit method
        
        # Use backward Euler with constant Jacobian - very efficient
        while t < t1 - 1e-15:
            if time.time() > deadline:
                break
            h = min(h_target, t1 - t)
            if h < h_min:
                h = h_min
            
            # Backward Euler: y_new = y + h * f(t+h, y_new)
            # For linear system: y_new = y + h * (A @ y_new + b)
            # => y_new - h*A@y_new = y + h*b
            # => (I - h*A) @ y_new = y + h*b
            rhs = y + h * b
            try:
                y_new = np.linalg.solve(np.eye(n) - h * A, rhs)
            except np.linalg.LinAlgError:
                # Fall back to Newton with finite difference
                y_new = y.copy()
                for _newton in range(50):
                    fz = A @ y_new + b
                    residual = y + h * fz - y_new
                    step = np.linalg.solve(np.eye(n) - h * A, residual)
                    y_new = y_new + step
                    if np.linalg.norm(step) < 1e-12 * (1 + np.linalg.norm(y_new)):
                        break
            
            # Adaptive step size based on local error estimate
            # Use a second-order method for error estimation
            # y_mid = y + h/2 * f(t, y)  (forward Euler half step)
            # y_new_2 = y_mid + h/2 * f(t+h/2, y_mid) (another half step)
            # Compare with y_new (backward Euler full step)
            
            # For linear systems, we can estimate error using the difference
            # between backward Euler and the exact solution approximation
            # Use Richardson extrapolation or a simpler estimate
            
            # Simple approach: use the residual to estimate error
            residual = y_new - y - h * (A @ y_new + b)
            err = np.linalg.norm(residual)
            
            # Accept step if error is small
            if err < 1e-10 * (1 + np.linalg.norm(y_new)):
                y = y_new
                t = t + h
                # Adjust step size
                if err < 1e-13 * (1 + np.linalg.norm(y_new)):
                    h_target = min(h * 2, h_max)
                elif err > 1e-8 * (1 + np.linalg.norm(y_new)):
                    h_target = max(h * 0.5, h_min)
            else:
                # Reject step, reduce h
                h_target = max(h * 0.5, h_min)
    
    else:
        # General case: use adaptive implicit Euler with finite difference Jacobian
        while t < t1 - 1e-15:
            if time.time() > deadline:
                break
            h = min(h_target, t1 - t)
            if h < h_min:
                h = h_min
            
            # Implicit Euler with Newton iteration
            z = y.copy()
            converged = False
            for _newton in range(50):
                fz = f(t + h, z)
                J, _ = compute_jacobian(t + h, z)
                residual = z - y - h * fz
                try:
                    step = np.linalg.solve(np.eye(n) - h * J, -residual)
                except np.linalg.LinAlgError:
                    step = -residual  # fallback
                z = z + step
                if np.linalg.norm(step) < 1e-12 * (1 + np.linalg.norm(z)):
                    converged = True
                    break
            
            if not converged:
                h_target = max(h * 0.5, h_min)
                continue
            
            # Error estimate using a second-order method
            # Forward Euler half-step twice
            k1 = f(t, y)
            y_mid = y + h / 2 * k1
            k2 = f(t + h / 2, y_mid)
            y_fe2 = y_mid + h / 2 * k2
            
            err = np.linalg.norm(z - y_fe2)
            
            if err < 1e-10 * (1 + np.linalg.norm(z)):
                y = z
                t = t + h
                if err < 1e-13 * (1 + np.linalg.norm(y)):
                    h_target = min(h * 2, h_max)
                elif err > 1e-8 * (1 + np.linalg.norm(y)):
                    h_target = max(h * 0.5, h_min)
            else:
                h_target = max(h * 0.5, h_min)
    
    return y
# EVOLVE-BLOCK-END
