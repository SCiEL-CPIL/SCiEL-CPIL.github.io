# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Use the same high-order integrator and tolerances as the reference solution
    # to minimize the relative difference.
    sol = solve_ivp(
        f,
        t_span,
        y0,
        method='DOP853',
        rtol=1e-11,
        atol=1e-13
    )
    
    if not sol.success:
        # Fallback: simple RK4 if solve_ivp fails (unlikely for this smooth ODE)
        dt = (t1 - t0) / 1000
        y = y0.copy()
        t = t0
        for _ in range(1000):
            k1 = f(t, y)
            k2 = f(t + dt/2, y + dt/2 * k1)
            k3 = f(t + dt/2, y + dt/2 * k2)
            k4 = f(t + dt, y + dt * k3)
            y = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += dt
        return y
        
    return sol.y[:, -1]
# EVOLVE-BLOCK-END
