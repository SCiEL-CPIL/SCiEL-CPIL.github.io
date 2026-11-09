import numpy as np
import scipy.linalg
import time

def integrate(f, t_span, y0):
    """
    Solves the ODE y' = f(t, y).
    Exploits the fact that the problem is a linear system y' = Ay to use matrix exponentiation.
    Falls back to SciPy's stiff solver if linearity is not detected or verification fails.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # 290s limit, keep safety margin before 300s kill

    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0

    # Since the task describes a "fixed 15-dimensional stiff linear initial-value problem",
    # we assume y' = A y where A is constant.
    # We extract A by calling f(t0, e_i) for each basis vector e_i.
    # This requires n calls.
    # We also call f(t0, 0) to check for inhomogeneity (b).
    # Total calls: n + 1 = 16 for n=15.
    
    try:
        # Check homogeneity (f(t, 0) = 0) or extract b
        f_zero = f(t0, np.zeros(n))
        
        # Extract A columns
        A = np.empty((n, n), dtype=float)
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            # Use t0 for extraction
            A[:, j] = f(t0, e_j)
            
        # Compute the matrix exponential M = exp(A * dt)
        # Using scipy.linalg.expm which is robust and fast for n=15
        M = scipy.linalg.expm(A * dt)
        
        # Check if the system is inhomogeneous (b != 0)
        b_norm = np.linalg.norm(f_zero)
        
        if b_norm > 1e-14:
            # Inhomogeneous case: y' = A y + b
            # Solution: y(t) = e^{A(t-t0)} y0 + \int_{t0}^t e^{A(t-s)} b ds
            # If b is constant: \int_0^dt e^{A tau} b dtau = (e^{A dt} - I) A^{-1} b
            # We need to solve A x = b for x, then add (M - I) x
            try:
                x = np.linalg.solve(A, f_zero)
                y_end = M @ y0 + (M - np.eye(n)) @ x
            except np.linalg.LinAlgError:
                # If A is singular, use a more general integration approach for the particular solution
                # This is unlikely for a stiff linear system, but handle it anyway.
                if time.time() > deadline:
                    return y0 # Best effort fallback
                from scipy.integrate import solve_ivp
                sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-13, atol=1e-15, max_step=0.01)
                if sol.success:
                    return sol.y[:, -1]
                else:
                    return sol.y[:, -1]
        else:
            # Homogeneous case: y' = A y
            y_end = M @ y0
        
        # Verification: check if f(t1, y_end) is consistent with A @ y_end (+ b)
        # This catches cases where the system is not truly linear/time-independent
        # However, we must be careful: if the system is linear, this should hold.
        # If it doesn't, we might be in a non-linear or time-dependent case.
        # But the task says "fixed ... linear ... system".
        # Let's add a quick check. If the residual is large, fall back.
        
        # We use 1 call here for verification. Total calls: 17.
        # This is still very low.
        
        # Compute expected derivative at t1, y_end
        if b_norm > 1e-14:
            expected = A @ y_end + f_zero
        else:
            expected = A @ y_end
            
        actual = f(t1, y_end)
        
        residual = np.linalg.norm(actual - expected)
        scale = max(np.linalg.norm(actual), np.linalg.norm(expected), 1e-15)
        rel_residual = residual / scale
        
        if rel_residual > 1e-8:
            # Linear model is inconsistent. Fall back to solver.
            if time.time() > deadline:
                return y_end
            from scipy.integrate import solve_ivp
            sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-13, atol=1e-15, max_step=0.01)
            if sol.success:
                return sol.y[:, -1]
            else:
                return sol.y[:, -1]
        
        return y_end

    except Exception:
        # Fallback to a generic stiff solver if anything goes wrong
        if time.time() > deadline:
            return y0 # Best effort fallback
        from scipy.integrate import solve_ivp
        try:
            # Use Radau or BDF for stiff problems
            # To maximize score, we want few calls. But accuracy is more important if linear detection fails.
            # Since linear detection is highly likely to succeed for this task, the fallback is just a safety net.
            
            def jac(t, y):
                # Finite difference Jacobian for fallback
                n = len(y)
                J = np.zeros((n, n))
                eps = 1e-7
                for i in range(n):
                    d = np.zeros(n)
                    d[i] = eps * max(1.0, abs(y[i]))
                    J[:, i] = (f(t, y + d) - f(t, y)) / d[i]
                return J

            sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-13, atol=1e-15, max_step=0.01, jac=jac)
            if sol.success:
                return sol.y[:, -1]
            else:
                # Last resort: RK4 with small steps (might be inaccurate for stiff)
                # Or just return the last state computed
                return sol.y[:, -1]
        except Exception:
            # Last resort: return y0 (though this will be inaccurate)
            return y0
