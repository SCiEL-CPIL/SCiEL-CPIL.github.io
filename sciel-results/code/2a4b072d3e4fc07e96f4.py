import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    y0_arr = np.array(y0, dtype=float)
    
    # If the problem is trivial or t_span is empty, handle it
    if t_span[0] == t_span[1]:
        return y0_arr

    calls = 0
    def f_counted(t, y):
        nonlocal calls
        calls += 1
        return f(t, y)
    
    try:
        sol = solve_ivp(
            f_counted, 
            t_span, 
            y0_arr, 
            method='DOP853', 
            rtol=1e-11, 
            atol=1e-13
        )
        
        if sol.success:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass

    # Fallback: Simple RK4 with many steps if solve_ivp fails
    # This is unlikely to be triggered
    t0, t1 = t_span
    y = y0_arr.copy()
    steps = 1000
    h = (t1 - t0) / steps
    t = t0
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y
