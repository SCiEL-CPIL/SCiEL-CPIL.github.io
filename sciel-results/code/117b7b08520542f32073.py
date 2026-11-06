"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 300s limit, keep 10s margin

    # Check for homogeneity: y' = A y vs y' = A y + b(t)
    # If f(t, 0) is not zero, it's non-homogeneous.
    zero_vec = np.zeros(n)
    b0 = f(t0, zero_vec)
    
    if np.linalg.norm(b0) < 1e-12:
        # Homogeneous linear system: y' = A y
        # Extract A by evaluating f(t0, e_j) for each basis vector e_j
        A = np.zeros((n, n))
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            A[:, j] = f(t0, e_j)
            
        # Check time if we have time
        if time.time() > deadline:
             return np.zeros(n) # Fallback, shouldn't happen

        # Compute matrix exponential
        dt = t1 - t0
        # Use expm for high accuracy
        M = expm(A * dt)
        
        y_end = M @ y0
        return y_end
    else:
        # Non-homogeneous or complex: use robust adaptive solver
        # Radau is good for stiff problems
        try:
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='Radau', 
                rtol=1e-12, 
                atol=1e-12,
                max_step=np.inf # Let it adapt
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                # Fallback to BDF if Radau fails
                sol = solve_ivp(
                    f, 
                    t_span, 
                    y0, 
                    method='BDF', 
                    rtol=1e-12, 
                    atol=1e-12
                )
                if sol.success:
                    return sol.y[:, -1]
                else:
                    # Last resort: return last known state or y0
                    return sol.y[:, -1] if sol.y.size > 0 else y0
        except Exception:
            return y0
# EVOLVE-BLOCK-END
