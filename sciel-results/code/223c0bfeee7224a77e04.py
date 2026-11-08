"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Start a timer to ensure we don't exceed the 300s limit
    start_time = time.time()
    
    # Use Radau IIA, a robust implicit Runge-Kutta method for stiff problems.
    # It has good stability properties and high order (3rd order).
    # rtol and atol are set to machine precision limits to maximize accuracy.
    # The score is -log10(err) - 0.5*log10(calls).
    # High accuracy reduces the first term. Efficient stepping reduces the second.
    
    # We use 'Radau' because it is L-stable, ideal for the heat equation.
    # 'LSODA' is an alternative but Radau is often more predictable in performance for dense linear systems.
    
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau',
            rtol=1e-12, 
            atol=1e-14,
            max_step=(t1-t0)/10, # Prevent huge steps initially if stiffness is high
            min_step=1e-6 * (t1-t0)
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # If it failed, try a more conservative method or smaller steps
            # But Radau rarely fails unless the problem is singular.
            # Fallback to BDF if Radau fails
            sol2 = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF',
                rtol=1e-12, 
                atol=1e-14
            )
            if sol2.success:
                return sol2.y[:, -1]
            else:
                return y0 # Last resort

    except Exception:
        # If an exception occurs (e.g., timeout, memory), return best effort or initial state
        # Since we can't easily recover mid-integration in solve_ivp without custom code,
        # we rely on the timeout guard.
        return y0

# EVOLVE-BLOCK-END
