import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrates the Lotka-Volterra system using DOP853 with tolerances
    matching the reference solution to minimize relative error.
    """
    sol = solve_ivp(
        f,
        t_span,
        y0,
        method='DOP853',
        rtol=1e-11,
        atol=1e-13
    )
    
    # If solve_ivp fails for some reason, fallback to a dense RK4
    # though this is extremely unlikely for this non-stiff problem.
    if not sol.success:
        # Fallback: RK4 with many steps
        t0, t1 = t_span
        y = np.array(y0, dtype=float)
        steps = 10000
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

    return sol.y[:, -1]
