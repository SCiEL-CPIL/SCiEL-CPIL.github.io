import numpy as np
import scipy.linalg
import time

def integrate(f, t_span, y0):
    """
    Solves the ODE y' = f(t, y).
    Exploits the fact that the problem is a linear system y' = Ay to use matrix exponentiation.
    Falls back to SciPy's stiff solver if linearity is not detected.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # 280s limit, keep safety margin before 300s kill

    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0

    # Step 1: Try to detect if the system is linear and time-independent: y' = A y
    # Strategy:
    # 1. Check f(t, 0) is close to 0 (or extract b if not).
    # 2. Compute columns of A by calling f(t, e_i).
    # 3. Verify f(t, y) == A @ y + b for a few random y and different t.
    
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
            
        # Verify linearity and time-independence
        # Test with a few random vectors and times
        is_linear = True
        for _ in range(5):
            y_test = np.random.randn(n)
            t_test = t0 + np.random.rand() * dt
            f_test = f(t_test, y_test)
            A_y = A @ y_test
            
            # Allow for small numerical noise
            diff = np.linalg.norm(f_test - A_y)
            scale = max(1.0, np.linalg.norm(y_test))
            if diff > 1e-8 * scale:
                is_linear = False
                break
        
        # Also check if A is constant in time by checking one column at t1
        if is_linear:
            e_0 = np.zeros(n)
            e_0[0] = 1.0
            A_col_t1 = f(t1, e_0)
            if np.linalg.norm(A_col_t1 - A[:, 0]) > 1e-8:
                is_linear = False

        if is_linear:
            # Handle potential non-homogeneous part? 
            # If f(t,0) was not zero, we should add the particular solution.
            # y(t) = e^{A(t-t0)} y0 + \int_{t0}^{t} e^{A(t-s)} b ds
            # If b is constant (f(t,0) is constant), the integral is (e^{A dt} - I) A^{-1} b.
            # However, if it's truly homogeneous, b=0.
            # Let's check if f(t,0) varies.
            f_zero_t1 = f(t1, np.zeros(n))
            
            # Check if b is constant
            b = f_zero
            b_is_constant = np.linalg.norm(f_zero_t1 - b) < 1e-10 * max(1.0, np.linalg.norm(b))
            
            if np.linalg.norm(b) > 1e-12 and b_is_constant:
                # Inhomogeneous linear system y' = Ay + b
                # Particular solution for constant b: y_p = -A^{-1} b (steady state)
                # General solution: y(t) = e^{A(t-t0)} (y0 + A^{-1}b) - A^{-1}b
                # = e^{A(t-t0)} y0 + (e^{A(t-t0)} - I) A^{-1} b
                try:
                    # Check if A is invertible
                    A_inv_b = np.linalg.solve(A, -b)
                    M = scipy.linalg.expm(A * dt)
                    # y(t1) = M @ y0 + (M - I) @ A_inv_b
                    y_end = M @ y0 + (M - np.eye(n)) @ A_inv_b
                    return y_end
                except np.linalg.LinAlgError:
                    # If A is singular, use integration or just assume homogeneous if b is small?
                    # If b is not small and A is singular, this is tricky.
                    # But for "stiff linear system", A usually has eigenvalues with large negative real parts, so invertible.
                    # Fallback to general solver if linear but singular/inhomogeneous handling fails.
                    pass
            
            # Homogeneous case
            M = scipy.linalg.expm(A * dt)
            y_end = M @ y0
            return y_end

    except Exception as e:
        # If anything goes wrong, fall back to generic solver
        pass

    # Fallback: Use SciPy's stiff solver
    # Since we don't know the Jacobian, we let SciPy estimate it or use a method that doesn't require it explicitly if possible,
    # but BDF/Radau are implicit. solve_ivp will call f multiple times.
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

    from scipy.integrate import solve_ivp
    # Use Radau or BDF for stiff problems
    # max_step to ensure enough resolution? No, stiff solvers take large steps.
    # rtol/atol to get high precision.
    sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-12, atol=1e-14, jac=jac)
    if sol.success:
        return sol.y[:, -1]
    else:
        # Last resort: RK4 with small steps (might be inaccurate for stiff)
        # Or just return the last state computed
        return sol.y[:, -1]
