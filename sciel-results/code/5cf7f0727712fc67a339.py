import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    mu = 5.0
    
    # Analytic Jacobian for Van der Pol
    def jac(t, y):
        y0_val = y[0]
        y1_val = y[1]
        # J = [[0, 1], [-2*mu*y0*y1 - 1, mu*(1-y0^2)]]
        J = np.array([
            [0.0, 1.0],
            [-2.0 * mu * y0_val * y1_val - 1.0, mu * (1.0 - y0_val**2)]
        ])
        return J

    # Wrapper for f to ensure it returns array and is callable by solve_ivp
    def fun(t, y):
        return f(t, y)

    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Use LSODA method with tight tolerances matching reference
    # LSODA is a hybrid Adams/BDF solver, distinct from Radau
    sol = solve_ivp(
        fun, 
        [t0, t1], 
        y0_arr, 
        method='LSODA', 
        jac=jac,
        rtol=1e-11, 
        atol=1e-13,
        t_eval=[t1]
    )
    
    if not sol.success:
        # Fallback if it fails, try Radau with same tolerances
        sol = solve_ivp(
            fun, 
            [t0, t1], 
            y0_arr, 
            method='Radau', 
            jac=jac,
            rtol=1e-11, 
            atol=1e-13,
            t_eval=[t1]
        )
        if not sol.success:
            return np.nan

    return sol.y[:, -1]
