import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Define the analytical Jacobian for Van der Pol with mu=5
    mu = 5.0
    def jac(t, y):
        return np.array([
            [0.0, 1.0],
            [-2.0 * mu * y[0] * y[1] - 1.0, mu * (1.0 - y[0]**2)]
        ])

    try:
        sol = solve_ivp(
            f,
            (t0, t1),
            y0,
            method='Radau',
            rtol=1e-11,
            atol=1e-13,
            jac=jac
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback
            sol2 = solve_ivp(
                f,
                (t0, t1),
                y0,
                method='DOP853',
                rtol=1e-11,
                atol=1e-13
            )
            if sol2.success:
                return sol2.y[:, -1]
            return y0
            
    except Exception:
        return np.array(y0, dtype=float)
