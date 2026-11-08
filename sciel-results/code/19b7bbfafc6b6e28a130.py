"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    deadline = time.time() + 280  # safety margin before 300s limit
    
    # Count f calls
    call_count = [0]
    def f_counted(t, y):
        call_count[0] += 1
        return f(t, y)
    
    # Try adaptive RK45 with tight tolerances
    from scipy.integrate import solve_ivp
    
    # Use DOP853 (high-order) for better accuracy per function call
    result = solve_ivp(
        f_counted, t_span, y0,
        method='DOP853',
        rtol=1e-12, atol=1e-14,
        max_step=(t1 - t0) / 500
    )
    
    if result.success and result.t[-1] == t1:
        return result.y[:, -1]
    
    # Fallback: try RK45 if DOP853 failed
    call_count[0] = 0
    result = solve_ivp(
        f_counted, t_span, y0,
        method='RK45',
        rtol=1e-10, atol=1e-12,
        max_step=(t1 - t0) / 200
    )
    
    if result.success:
        return result.y[:, -1]
    
    # Last resort: fixed-step RK4
    h = (t1 - t0) / 5000
    t, y = t0, y0.copy()
    for _ in range(5000):
        if time.time() > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2*k1)
        k3 = f(t + h/2, y + h/2*k2)
        k4 = f(t + h, y + h*k3)
        y = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y
# EVOLVE-BLOCK-END
