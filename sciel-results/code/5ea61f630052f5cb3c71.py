"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system using adaptive DOP853 with no step limit,
    allowing the solver to choose optimal step sizes for minimal function calls.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 280.0
    
    try:
        # Use DOP853 with tight tolerances and no max_step restriction
        # This allows the solver to take larger steps in smooth regions
        sol = solve_ivp(
            f, 
            t_span, 
            y,
            method='DOP853',
            rtol=1e-12,
            atol=1e-14,
            max_step=np.inf
        )
        
        if sol.success:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)) and y_end.shape == y0.shape:
                return y_end
    except Exception:
        pass
    
    # Fallback: use solve_ivp with default settings
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y,
            method='RK45',
            rtol=1e-10,
            atol=1e-12
        )
        if sol.success:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)) and y_end.shape == y0.shape:
                return y_end
    except Exception:
        pass
    
    # Last resort: simple Euler with many steps (should not be reached)
    n_steps = 10000
    h = (t1 - t0) / n_steps
    t = t0
    y = np.array(y0, dtype=float)
    
    for _ in range(n_steps):
        if time.time() > deadline:
            break
        y = y + h * f(t, y)
        t += h
        
    return y

# EVOLVE-BLOCK-END
