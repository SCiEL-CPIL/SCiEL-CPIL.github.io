import numpy as np
from scipy.linalg import expm, inv
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # 1. Probe the system to see if it is linear with constant coefficients: y' = A*y + b
    # Evaluate b = f(t0, 0)
    y_zero = np.zeros(n)
    b = np.array(f(t0, y_zero), dtype=float)
    
    # Evaluate A columns: A[:, j] = f(t0, e_j)
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = np.array(f(t0, e_j), dtype=float)
        
    # Check if the system is consistent with y' = A*y + b at a different point
    # Use a random test point to verify linearity and constant coefficients
    # We pick a simple test point, e.g., y_test = ones
    y_test = np.ones(n)
    t_test = t0 + 0.5 * (t1 - t0)
    f_test = np.array(f(t_test, y_test), dtype=float)
    f_pred = A @ y_test + b
    
    # If the prediction is close, we assume constant linear system
    # Tolerance for this check should be tight
    if np.allclose(f_test, f_pred, rtol=1e-10, atol=1e-12):
        # System is y' = A*y + b with constant A, b
        # Solution: y(t) = e^{A(t-t0)} y0 + \int_{t0}^t e^{A(t-\tau)} b d\tau
        # For constant b, the integral is (e^{A(t-t0)} - I) A^{-1} b
        
        dt = t1 - t0
        M = expm(A * dt)
        
        # Check if b is effectively zero
        if np.linalg.norm(b) < 1e-15:
            y_end = M @ y0
        else:
            # Compute A^{-1} b. If A is singular, use pinv.
            # In stiff linear systems, A is rarely singular in a way that breaks this, 
            # but pinv is safer.
            try:
                A_inv_b = inv(A) @ b
            except np.linalg.LinAlgError:
                A_inv_b = np.linalg.pinv(A) @ b
            
            # Integral term: (M - I) @ A_inv_b
            y_end = M @ y0 + (M - np.eye(n)) @ A_inv_b
        
        return y_end

    else:
        # Fallback: Use Radau solver with Jacobian
        # Define Jacobian function. Since we don't know the exact form, 
        # we use finite differences, but we can hint the solver.
        # However, if it's not constant linear, we just let solve_ivp handle it.
        # To save calls, we can try to use the A we found if it's close, 
        # but if the check failed, A might be time-dependent or the system non-linear.
        # Safest is to let solve_ivp use finite differences for J, or 
        # provide a J that uses the same finite difference logic.
        
        def jac(t, y):
            # Finite difference Jacobian
            J = np.zeros((n, n))
            for j in range(n):
                e = np.zeros(n)
                e[j] = 1e-8 * max(1.0, abs(y[j]))
                J[:, j] = (f(t, y + e) - f(t, y)) / e[j]
            return J

        sol = solve_ivp(f, t_span, y0, method='Radau', jac=jac, rtol=1e-12, atol=1e-14)
        if not sol.success:
             # Fallback to BDF if Radau fails
             sol = solve_ivp(f, t_span, y0, method='BDF', jac=jac, rtol=1e-12, atol=1e-14)
        return sol.y[:, -1]
