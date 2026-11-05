"""
Rewritten program: ODE solver for 15-dim stiff linear system.
Strategy:
1. Detect if the system matrix A is time-invariant by sampling f at t0 and t1.
2. If invariant, compute the exact solution using matrix exponential (expm).
3. If time-varying, use a high-accuracy implicit solver (Radau) with tight tolerances.
"""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg as sla
import scipy.integrate as sci

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Helper to get matrix A at a given time t
    def get_A(t):
        A = np.zeros((n, n))
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            # f(t, y) = A(t) @ y
            A[:, j] = f(t, e_j)
        return A

    # Sample A at start and end
    A_start = get_A(t0)
    A_end = get_A(t1)
    
    # Check for time-invariance
    # Use Frobenius norm
    diff_norm = np.linalg.norm(A_start - A_end, 'fro')
    scale = max(1.0, np.linalg.norm(A_start, 'fro'), np.linalg.norm(A_end, 'fro'))
    
    if diff_norm < 1e-12 * scale:
        # System is effectively time-invariant: y' = A y
        dt = t1 - t0
        # Compute matrix exponential
        # expm is robust for 15x15 matrices
        M = sla.expm(dt * A_start)
        y_end = M @ y0
        return y_end
    else:
        # System is time-varying: y' = A(t) y
        # Use Radau method which is suitable for stiff problems
        # Tight tolerances to maximize accuracy
        # We assume the cost of f is low enough that Radau won't hit the 2M call limit
        # for a 15-dim problem on [0,1] with these tolerances.
        
        # Try Radau first
        sol = sci.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-14,
            # max_step=np.inf # default
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback to BDF if Radau fails (e.g. due to singularities or step size issues)
            sol = sci.solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                rtol=1e-10, 
                atol=1e-12
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                # Last resort: RK45 with high steps (unlikely to be needed)
                sol = sci.solve_ivp(
                    f, 
                    t_span, 
                    y0, 
                    method='RK45', 
                    rtol=1e-8, 
                    atol=1e-10
                )
                return sol.y[:, -1]
# EVOLVE-BLOCK-END
