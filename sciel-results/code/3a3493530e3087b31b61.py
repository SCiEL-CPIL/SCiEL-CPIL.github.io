"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Determine the linear structure: f(t, y) = A @ y + b(t)
    # Use a few calls to determine A and b at t0
    # First, get b(t0) by calling f with y=0
    y_zero = np.zeros(n)
    f0 = np.array(f(t0, y_zero), dtype=float)  # This is b(t0)
    
    # Determine A by perturbing each basis direction
    A = np.zeros((n, n))
    eps = 1e-8
    for j in range(n):
        e = np.zeros(n)
        e[j] = eps
        fj = np.array(f(t0, e), dtype=float)
        A[:, j] = (fj - f0) / eps

    # Define the Jacobian function
    # Since the system is linear y' = A*y + b(t), the Jacobian is constant A
    def jac(t, y):
        return A

    # Solve using a stiff solver
    # Radau is good for stiff problems
    # We use tight tolerances to minimize error
    # The cost is the number of calls to f
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='Radau',
        jac=jac,
        rtol=1e-12,
        atol=1e-12,
        max_step=0.05  # Limit step size to ensure accuracy, but not too small
    )
    
    if not sol.success:
        # Fallback if Radau fails, try BDF
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF',
            jac=jac,
            rtol=1e-12,
            atol=1e-12
        )
        
    return sol.y[:, -1]
# EVOLVE-BLOCK-END
