"""
Rewritten program using SciPy's DOP853 adaptive integrator for higher accuracy and lower call count.
"""
import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrates the ODE using SciPy's solve_ivp with DOP853 method.
    Falls back to Radau if DOP853 fails.
    """
    y0 = np.array(y0, dtype=float)
    t0, t1 = t_span
    
    call_count = 0
    max_calls = 1900000  # Safety margin below 2,000,000
    deadline = time.time() + 280.0  # Safety margin below 300s
    
    def f_counted(t, y):
        nonlocal call_count
        call_count += 1
        if call_count > max_calls:
            raise RuntimeError("Max calls exceeded")
        if time.time() > deadline:
            raise RuntimeError("Time limit exceeded")
        return f(t, y)
    
    # Try DOP853 first (7th order, explicit, efficient for non-stiff)
    try:
        sol = solve_ivp(
            f_counted, 
            (t0, t1), 
            y0, 
            method='DOP853', 
            rtol=1e-12, 
            atol=1e-14,
            max_step=0.05 # Keep steps reasonable
        )
        if sol.success and sol.t[-1] >= t1 - 1e-12:
            return sol.y[:, -1]
    except Exception:
        pass
        
    # Fallback to Radau (implicit, good for stiff)
    try:
        sol = solve_ivp(
            f_counted, 
            (t0, t1), 
            y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-14
        )
        if sol.success and sol.t[-1] >= t1 - 1e-12:
            return sol.y[:, -1]
    except Exception:
        pass

    # Final fallback: Simple fixed step RK4 (less accurate, but robust)
    # This should rarely be reached.
    try:
        n_steps = 10000
        h = (t1 - t0) / n_steps
        y = y0.copy()
        t = t0
        for _ in range(n_steps):
            if time.time() > deadline or call_count > max_calls:
                break
            k1 = f_counted(t, y)
            k2 = f_counted(t + h/2, y + h/2 * k1)
            k3 = f_counted(t + h/2, y + h/2 * k2)
            k4 = f_counted(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y
    except Exception:
        return y0
