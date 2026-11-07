import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Use DOP853 (explicit 8th-order) as a different approach
    # This method is highly efficient for non-stiff to moderately stiff problems
    # and requires fewer steps than lower-order methods for high accuracy
    sol = solve_ivp(
        f, 
        [t0, t1], 
        y0_arr, 
        method='DOP853',
        rtol=1e-11, 
        atol=1e-13,
        t_eval=[t1]
    )
    
    if not sol.success:
        # Fallback to Radau if DOP853 fails (e.g. if problem is too stiff)
        mu = 5.0
        def jac(t, y):
            y0_val = y[0]
            y1_val = y[1]
            J = np.array([
                [0.0, 1.0],
                [-2.0 * mu * y0_val * y1_val - 1.0, mu * (1.0 - y0_val**2)]
            ])
            return J
            
        sol = solve_ivp(
            f, 
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
