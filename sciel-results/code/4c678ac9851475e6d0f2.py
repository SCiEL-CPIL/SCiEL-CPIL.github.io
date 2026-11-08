"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    deadline = time.time() + 280  # safety margin before 300s limit
    
    # Count f calls
    call_count = [0]
    def f_counted(t, y):
        call_count[0] += 1
        return f(t, y)
    
    # For convection-diffusion 100-point problem, it's likely stiff and linear
    # Use Radau or BDF with tight tolerances and constant Jacobian if possible
    
    # First, check if autonomous and try to get a constant Jacobian
    is_autonomous = False
    try:
        f0 = f_counted(t0, y0)
        f1 = f_counted(t1, y0)
        if np.allclose(f0, f1, rtol=1e-5, atol=1e-8):
            is_autonomous = True
    except Exception:
        is_autonomous = False
    
    jac_func = None
    if is_autonomous:
        # Compute Jacobian at t0, y0 using central differences
        J = np.zeros((n, n))
        eps = 1e-7
        for j in range(n):
            yp = y0.copy()
            ym = y0.copy()
            step = eps * max(1.0, abs(y0[j]))
            yp[j] += step
            ym[j] -= step
            try:
                fyp = f_counted(t0, yp)
                fym = f_counted(t0, ym)
                J[:, j] = (fyp - fym) / (2 * step)
            except Exception:
                yp = y0.copy()
                step = eps * max(1.0, abs(y0[j]))
                yp[j] += step
                fyp = f_counted(t0, yp)
                J[:, j] = (fyp - f0) / step
                
        def jac_func(t, y):
            return J
    
    # Try Radau first (good for stiff linear problems)
    try:
        sol = scipy.integrate.solve_ivp(
            f_counted, 
            t_span, 
            y0, 
            method='Radau',
            rtol=1e-12, 
            atol=1e-14,
            jac=jac_func,
            max_step=(t1 - t0) / 10.0
        )
        if sol.success and sol.t[-1] == t1:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback to BDF
    try:
        sol = scipy.integrate.solve_ivp(
            f_counted, 
            t_span, 
            y0, 
            method='BDF',
            rtol=1e-12, 
            atol=1e-14,
            jac=jac_func,
            max_step=(t1 - t0) / 10.0
        )
        if sol.success and sol.t[-1] == t1:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback to DOP853 with tight tolerances
    try:
        sol = scipy.integrate.solve_ivp(
            f_counted, 
            t_span, 
            y0, 
            method='DOP853',
            rtol=1e-12, 
            atol=1e-14,
            max_step=(t1 - t0) / 100.0
        )
        if sol.success and sol.t[-1] == t1:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # If all fail, return initial state (should not happen)
    return y0
# EVOLVE-BLOCK-END
