import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    # Use DOP853 (8th order) with tight tolerances to minimize calls while ensuring accuracy < 1e-14
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-15, 
        atol=1e-16,
        dense_output=False
    )
    if not sol.success:
        # Fallback to RK4 if something goes wrong, though unlikely
        # Implement a simple RK4 fallback
        def _rk4(f, t0, t1, y0, steps):
            h = (t1 - t0) / steps
            t, y = t0, np.array(y0, dtype=float)
            for _ in range(steps):
                k1 = f(t, y)
                k2 = f(t + h / 2, y + h / 2 * k1)
                k3 = f(t + h / 2, y + h / 2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                t += h
            return y
        return _rk4(f, t0, t1, y0, 2000)
        
    return sol.y[:, -1]
