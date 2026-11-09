"""ODE solver: 15-dimensional stiff linear system using matrix extraction and exponential integrator."""
import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    deadline = time.time() + 280.0  # safety margin before 300s limit
    
    # Step 1: Extract the linear system matrix A by probing f
    # For a linear system y' = A*y + b(t), we can extract A by evaluating f at different points
    # f(t, y) = A @ y + b(t)
    # f(t, y1) - f(t, y2) = A @ (y1 - y2)
    
    # Probe at t0 with perturbed states to extract A
    # Use a set of basis vectors to determine A
    A = np.zeros((n, n), dtype=float)
    
    # First, get f(t0, y0) as baseline
    f0 = f(t0, y0)
    
    # Probe each column of A
    for j in range(n):
        # Perturb the j-th component
        delta = 1e-7 * max(1.0, abs(y0[j]))
        y_pert = y0.copy()
        y_pert[j] += delta
        f_pert = f(t0, y_pert)
        A[:, j] = (f_pert - f0) / delta
    
    # Check if the system is truly linear by verifying f(t, y) = A@y + b(t)
    # Sample at a few points to verify linearity
    is_linear = True
    for _ in range(3):
        t_test = t0 + (t1 - t0) * np.random.uniform(0.1, 0.9)
        y_test = y0 + np.random.randn(n) * 0.1
        f_test = f(t_test, y_test)
        # If linear with constant A, f(t_test, y_test) should be consistent
        # But b(t) may vary, so we can't directly check without knowing b(t)
        # Instead, check that the Jacobian is constant by probing at another point
        if time.time() > deadline - 10:
            break
    
    # Verify A is constant by probing at a different t
    t_check = (t0 + t1) / 2
    f_check0 = f(t_check, y0)
    A_check = np.zeros((n, n), dtype=float)
    for j in range(n):
        delta = 1e-7 * max(1.0, abs(y0[j]))
        y_pert = y0.copy()
        y_pert[j] += delta
        f_pert = f(t_check, y_pert)
        A_check[:, j] = (f_pert - f_check0) / delta
    
    # If A is approximately the same at different t, the system has constant A
    A_diff = np.max(np.abs(A - A_check))
    if A_diff < 1e-4:
        # Use the average A for better accuracy
        A = (A + A_check) / 2.0
        is_linear = True
    else:
        is_linear = False
    
    if is_linear:
        # Use matrix exponential for exact solution of y' = A*y + b(t)
        # But b(t) may be non-zero and time-dependent
        # For pure linear y' = A*y, solution is y(t) = exp(A*(t-t0)) @ y0
        # For y' = A*y + b(t), we need to integrate b(t)
        
        # First, check if b(t) is zero by evaluating f at y=0
        y_zero = np.zeros(n, dtype=float)
        f_zero = f(t0, y_zero)
        b_norm = np.linalg.norm(f_zero)
        
        if b_norm < 1e-10:
            # Pure linear system: y' = A*y
            # Solution: y(t1) = expm(A * (t1-t0)) @ y0
            from scipy.linalg import expm
            dt = t1 - t0
            M = expm(A * dt)
            y_final = M @ y0
            return y_final
        else:
            # System with forcing: y' = A*y + b(t)
            # Use a high-order implicit method with the known A
            # Since we know A, we can solve the implicit system efficiently
            
            # Use a BDF method with the known Jacobian
            # For stiff linear systems, this will be very accurate
            
            # Use scipy's solve_ivp with the known Jacobian
            from scipy.integrate import solve_ivp
            
            def f_wrapper(t, y):
                return f(t, y)
            
            def jac_wrapper(t, y):
                return A
            
            # Use Radau method which is good for stiff problems
            sol = solve_ivp(f_wrapper, t_span, y0, method='Radau', 
                           jac=jac_wrapper, rtol=1e-12, atol=1e-14,
                           max_step=(t1-t0)/100)
            
            if sol.success:
                return sol.y[:, -1]
            else:
                # Fallback to matrix exponential approach with quadrature for b(t)
                # y(t) = expm(A*(t-t0))@y0 + integral_0^t expm(A*(t-s))@b(s) ds
                # Approximate the integral with high-order quadrature
                from scipy.linalg import expm
                from scipy.integrate import quad_vec
                
                dt = t1 - t0
                M_final = expm(A * dt)
                
                # Integrate the forcing term
                def integrand(s):
                    M_s = expm(A * (t1 - s))
                    b_s = f(s, np.zeros(n, dtype=float))
                    return M_s @ b_s
                
                # Use adaptive quadrature
                result, err = quad_vec(integrand, t0, t1, epsabs=1e-14, epsrel=1e-12)
                y_final = M_final @ y0 + result
                return y_final
    else:
        # Non-linear or time-varying A: use a robust stiff solver
        from scipy.integrate import solve_ivp
        
        def f_wrapper(t, y):
            return f(t, y)
        
        # Use Radau with adaptive Jacobian
        sol = solve_ivp(f_wrapper, t_span, y0, method='Radau', 
                       rtol=1e-12, atol=1e-14,
                       max_step=(t1-t0)/100)
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback to explicit method with small steps
            h = (t1 - t0) / 10000
            t, y = t0, y0.copy()
            for _ in range(10000):
                if time.time() > deadline:
                    break
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2*k1)
                k3 = f(t + h/2, y + h/2*k2)
                k4 = f(t + h, y + h*k3)
                y = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
                t += h
            return y
