"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Extract Jacobian A assuming f(t,y) = A*y + b(t)
    # This is valid for the "stiff linear system" description.
    # We do this once to avoid finite difference costs in Radau.
    y_zero = np.zeros(n)
    f0 = np.array(f(t0, y_zero), dtype=float)
    
    A = np.zeros((n, n))
    eps = 1e-8
    for j in range(n):
        e = np.zeros(n)
        e[j] = eps
        fj = np.array(f(t0, e), dtype=float)
        A[:, j] = (fj - f0) / eps
        
    def jac(t, y):
        return A
        
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='Radau', 
        jac=jac,
        rtol=1e-12, 
        atol=1e-12,
        t_eval=[t1]
    )
    
    if not sol.success:
        # Fallback to BDF if Radau fails
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF', 
            jac=jac,
            rtol=1e-12, 
            atol=1e-12,
            t_eval=[t1]
        )
        
    return sol.y[:, -1]
# EVOLVE-BLOCK-END
