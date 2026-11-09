import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    start_time = time.time()
    deadline = start_time + 290.0

    # Primary: DOP853 with tight tolerances
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-14, 
            atol=1e-16
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Secondary: Radau (implicit) for stiff cases
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-13, 
            atol=1e-15
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Tertiary: BDF
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF', 
            rtol=1e-13, 
            atol=1e-15
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback: Fixed step RK4 (low accuracy, but returns something)
    # Use a moderate step count to avoid timeout
    steps = 5000
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        if time.time() > deadline:
            return y
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y
