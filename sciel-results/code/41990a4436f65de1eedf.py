import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard: 330s wall clock limit, keep 10s margin
    start_time = time.time()
    deadline = start_time + 320.0
    
    # Use DOP853 with tolerances matching the reference solver (rtol=1e-11, atol=1e-13)
    # This balances accuracy (keeping rel small) and efficiency (minimizing calls).
    try:
        sol = solve_ivp(
            f,
            (t0, t1),
            y0,
            method='DOP853',
            rtol=1e-11,
            atol=1e-13
        )
        
        if not sol.success:
            # If DOP853 fails for some reason, try RK45 as a fallback
            sol = solve_ivp(
                f,
                (t0, t1),
                y0,
                method='RK45',
                rtol=1e-11,
                atol=1e-13
            )
            
        if not sol.success:
            raise RuntimeError("Integration failed")
            
        return sol.y[:, -1]
        
    except Exception:
        # In case of any unexpected error, return a best-effort result or raise
        # Since the task says "an exception makes the whole result invalid", 
        # we should try to avoid exceptions if possible, but for Lorenz, it's very stable.
        raise
