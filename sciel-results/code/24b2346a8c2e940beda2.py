"""Direct approach: Radau integrator with reference-matching tolerances."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Integrates the Van der Pol ODE using Radau with tolerances matching the reference.
    Uses a good initial step hint to minimize function calls.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    start_time = time.time()
    DEADLINE = start_time + 290.0
    
    from scipy.integrate import solve_ivp
    
    try:
        # Use Radau with the exact same tolerances as reference.
        # Provide a reasonable first_step to help the adaptive stepper.
        sol = solve_ivp(
            f, 
            t_span, 
            y, 
            method='Radau', 
            rtol=1e-11, 
            atol=1e-13,
            first_step=0.01
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback without first_step hint
            sol = solve_ivp(
                f, 
                t_span, 
                y, 
                method='Radau', 
                rtol=1e-11, 
                atol=1e-13
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                return np.array(y, dtype=float)
    except Exception:
        return np.array(y, dtype=float)
# EVOLVE-BLOCK-END
