import numpy as np
import scipy.linalg
import scipy.integrate
import time

def integrate(f, t_span, y0):
    t_start = time.time()
    deadline = t_start + 285.0
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = y0.shape[0]

    # Reconstruct A and b
    zero = np.zeros(n)
    b = f(t0, zero)
    
    A = np.zeros((n, n))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1.0
        fi = f(t0, ei)
        A[:, i] = fi - b

    # Strategy: Use solve_ivp with Radau for high accuracy on stiff systems.
    # This might use more calls, but if the expm error was significant, 
    # the accuracy gain should outweigh the call penalty.
    
    def ode_func(t, y):
        return A @ y + b

    # Use Radau, which is good for stiff problems.
    # Set tight tolerances to maximize accuracy.
    # max_step is left to the solver, but we can set a cap if needed.
    try:
        sol = scipy.integrate.solve_ivp(
            ode_func, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-10, 
            atol=1e-10,
            max_step=0.01 # Ensure it doesn't take huge steps in stiff regions
        )
        if sol.success:
            y_final_ivp = sol.y[:, -1]
        else:
            # Fallback to expm if solver fails
            M = np.zeros((n+1, n+1))
            M[:n, :n] = A
            M[:n, n] = b
            dt = t1 - t0
            M_scaled = dt * M
            expM = scipy.linalg.expm(M_scaled)
            z_init = np.zeros(n+1)
            z_init[:n] = y0
            z_init[n] = 1.0
            z_final = expM @ z_init
            y_final_ivp = z_final[:n]
            
    except Exception:
        # Fallback to expm on exception
        M = np.zeros((n+1, n+1))
        M[:n, :n] = A
        M[:n, n] = b
        dt = t1 - t0
        M_scaled = dt * M
        expM = scipy.linalg.expm(M_scaled)
        z_init = np.zeros(n+1)
        z_init[:n] = y0
        z_init[n] = 1.0
        z_final = expM @ z_init
        y_final_ivp = z_final[:n]

    return y_final_ivp
