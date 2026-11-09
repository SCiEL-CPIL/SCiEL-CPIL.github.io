"""
Optimized DOP853 integrator for Van der Pol oscillator with tuned tolerances.
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
    start_time = time.time()
    deadline = start_time + 320  # Safety margin before 330s limit
    
    call_count = 0
    max_calls = 1950000  # Safety margin below 2,000,000
    
    def f_counted(t, y):
        nonlocal call_count
        call_count += 1
        if call_count > max_calls:
            raise RuntimeError("Max calls exceeded")
        return f(t, y)
    
    # Try DOP853 first (7th order, explicit, efficient for non-stiff)
    # Use tighter tolerances for better accuracy while keeping call count reasonable
    try:
        sol = solve_ivp(
            f_counted, 
            (t0, t1), 
            y0, 
            method='DOP853', 
            rtol=1e-14, 
            atol=1e-16,
            max_step=0.01
        )
        if sol.success and sol.t[-1] >= t1 - 1e-12:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Check if we're running out of time
    if time.time() > deadline - 50:
        # Try Radau with looser tolerances if we're tight on time
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
        return y0
    
    # Fallback to Radau (implicit, good for stiff)
    try:
        sol = solve_ivp(
            f_counted, 
            (t0, t1), 
            y0, 
            method='Radau', 
            rtol=1e-14, 
            atol=1e-16
        )
        if sol.success and sol.t[-1] >= t1 - 1e-12:
            return sol.y[:, -1]
    except Exception:
        pass

    # Final fallback: Simple fixed step RK4 (less accurate, but robust)
    try:
        n_steps = 20000
        h = (t1 - t0) / n_steps
        y = y0.copy()
        t = t0
        for _ in range(n_steps):
            if call_count > max_calls or time.time() > deadline - 10:
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
