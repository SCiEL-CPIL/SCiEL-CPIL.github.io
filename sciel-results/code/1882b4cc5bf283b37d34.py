"""
Solves a 15-dimensional stiff linear ODE system y' = A y.
Exploits the linear constant-coefficient structure to compute the exact solution
via matrix exponential, minimizing function calls to ~15.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.linalg as sla
import scipy.integrate as ssi

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin for 300s limit
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # 1. Determine the system matrix A by probing the basis.
    # We assume f(t, y) = A y.
    # We evaluate f at standard basis vectors to get columns of A.
    A = np.zeros((n, n), dtype=float)
    
    # Check linearity and time-independence with a few probes
    # Probe 1: y0
    f_y0 = f(t0, y0)
    
    # If the system is linear and constant, f(t, y) should be A y for any t.
    # Let's estimate A using the basis vectors.
    for j in range(n):
        e_j = np.zeros(n, dtype=float)
        e_j[j] = 1.0
        # Use t0 for probing. If the system is time-invariant, this is sufficient.
        A[:, j] = f(t0, e_j)
        
        # Check deadline
        if time.time() > deadline:
             # Fallback to a simple solver if we run out of time (unlikely)
             sol = ssi.solve_ivp(f, [t0, t1], y0, method='Radau', rtol=1e-10, atol=1e-12)
             return sol.y[:, -1]

    # Verify the model: Check if f(t1, y0) matches A @ y0
    # This also checks for time-dependence.
    f_check = f(t1, y0)
    y_model = A @ y0
    
    # Calculate relative difference
    diff = np.linalg.norm(f_check - y_model)
    scale = max(1.0, np.linalg.norm(y_model))
    
    if diff > 1e-8 * scale:
        # The system might be time-dependent or non-linear.
        # However, the task specifies "fixed 15-dimensional stiff linear system".
        # If it's linear but time-dependent y' = A(t)y, we can't use a single matrix exponential.
        # Given the "stiff linear system" description in such tasks usually implies constant coefficients 
        # or a specific structure that allows reduction. 
        # If the check fails significantly, fall back to a robust implicit solver.
        sol = ssi.solve_ivp(f, [t0, t1], y0, method='Radau', rtol=1e-12, atol=1e-14)
        if sol.success:
            return sol.y[:, -1]
        else:
            # Last resort
            sol = ssi.solve_ivp(f, [t0, t1], y0, method='BDF', rtol=1e-12, atol=1e-14)
            return sol.y[:, -1]
            
    # 2. Compute the matrix exponential
    # Solution is y(t1) = exp((t1 - t0) * A) @ y0
    dt = t1 - t0
    try:
        M = sla.expm(dt * A)
    except Exception:
        # Fallback if expm fails (e.g., numerical issues)
        sol = ssi.solve_ivp(f, [t0, t1], y0, method='Radau', rtol=1e-12, atol=1e-14)
        return sol.y[:, -1]
        
    y_final = M @ y0
    
    # 3. Verify the result by checking the derivative at the end?
    # Or just return it. The matrix exponential is exact for constant linear systems.
    
    return y_final
# EVOLVE-BLOCK-END
