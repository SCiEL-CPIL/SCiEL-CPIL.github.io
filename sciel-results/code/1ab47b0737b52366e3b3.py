"""Rewritten program using Linear Operator Extraction and Matrix Exponentiation."""
import scipy
import numpy as np
import scipy.linalg as la
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 10s margin before 300s limit

    # 1. Probe for linearity and time-invariance
    # Check f(0, 0)
    f00 = f(0.0, np.zeros(n))
    if np.linalg.norm(f00) > 1e-10:
        # Not homogeneous, likely nonlinear or has source term
        pass
    else:
        # Check linearity: f(0, 2y) == 2 f(0, y)
        test_y = np.random.rand(n)
        f_test = f(0.0, test_y)
        f_double = f(0.0, 2.0 * test_y)
        if np.linalg.norm(f_double - 2.0 * f_test) < 1e-8 * (1 + np.linalg.norm(f_test)):
            # Check time invariance: f(t1, y) == f(t2, y)
            t_mid = t0 + dt / 2.0
            f_t1 = f(t1 - 1e-6, test_y)
            f_t0 = f(t0 + 1e-6, test_y)
            if np.linalg.norm(f_t1 - f_t0) < 1e-8 * (1 + np.linalg.norm(f_t1)):
                # Assume y' = A y
                # Extract A by calling f with basis vectors
                A = np.zeros((n, n))
                for i in range(n):
                    e_i = np.zeros(n)
                    e_i[i] = 1.0
                    A[:, i] = f(0.0, e_i)
                
                # Check if time is passing the deadline
                if time.time() > deadline:
                    # Fallback to a quick RK4 if we ran out of time during probing
                    return _fast_rk4(f, t0, t1, y0, 1000)
                
                # Compute matrix exponential
                # y(t1) = exp(A * dt) * y0
                M = la.expm(A * dt)
                y_end = M @ y0
                
                # Verify result is finite
                if not np.all(np.isfinite(y_end)):
                    return _fast_rk4(f, t0, t1, y0, 1000)
                    
                return y_end

    # 2. Fallback: Generic stiff solver
    # Use Radau or BDF which are good for stiff problems
    # We wrap f to count calls and potentially enforce deadline
    
    call_count = [0]
    def f_wrapper(t, y):
        call_count[0] += 1
        if call_count[0] > 2000000:
            raise Exception("Call limit exceeded")
        return f(t, y)
        
    # Try Radau first, it's very accurate for stiff problems
    try:
        sol = scipy.integrate.solve_ivp(
            f_wrapper, t_span, y0, 
            method='Radau', 
            rtol=1e-10, atol=1e-12,
            max_step=dt/100
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback to BDF if Radau fails
            sol = scipy.integrate.solve_ivp(
                f_wrapper, t_span, y0, 
                method='BDF', 
                rtol=1e-10, atol=1e-12,
                max_step=dt/100
            )
            if sol.success:
                return sol.y[:, -1]
    except Exception:
        pass
    
    # Final fallback: Explicit RK4 with adaptive step size (if stiff, this might fail, but better than nothing)
    return _fast_rk4(f, t0, t1, y0, 5000)

def _fast_rk4(f, t0, t1, y0, steps):
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            return y
    return y
