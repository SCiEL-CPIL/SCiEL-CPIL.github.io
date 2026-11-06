"""
Solves a 15-dimensional stiff linear ODE system y' = A y.
Uses the matrix exponential for the exact solution.
Minimizes function calls to exactly n (dimension of the system).
"""
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
    # We assume f(t, y) = A y (linear, time-invariant).
    # This requires n function calls, one for each column of A.
    A = np.zeros((n, n), dtype=float)
    
    for j in range(n):
        e_j = np.zeros(n, dtype=float)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)
        
        # Check deadline
        if time.time() > deadline:
             # Fallback if probing takes too long (unlikely for n=15)
             sol = ssi.solve_ivp(f, [t0, t1], y0, method='Radau', rtol=1e-10, atol=1e-12)
             return sol.y[:, -1]

    # 2. Compute the matrix exponential
    dt = t1 - t0
    
    try:
        # Use standard expm which is robust and uses Schur decomposition internally
        M = sla.expm(dt * A)
    except Exception:
        # Fallback to numerical integration if expm fails
        sol = ssi.solve_ivp(f, [t0, t1], y0, method='Radau', rtol=1e-12, atol=1e-14)
        if sol.success:
            return sol.y[:, -1]
        else:
            sol = ssi.solve_ivp(f, [t0, t1], y0, method='BDF', rtol=1e-12, atol=1e-14)
            return sol.y[:, -1]
            
    y_final = M @ y0
    
    return y_final
