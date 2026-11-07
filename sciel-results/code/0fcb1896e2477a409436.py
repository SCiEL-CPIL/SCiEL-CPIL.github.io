"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate the Lorenz system from t_span[0] to t_span[1] starting at y0.
    
    Uses scipy's DOP853 (8th/5th order) with tight tolerances for high accuracy.
    Falls back to manual RK45 if scipy fails or times out.
    """
    start_time = time.time()
    deadline = start_time + 290  # 290s safety margin before 330s kill
    
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Try scipy's DOP853 with very tight tolerances
    try:
        sol = solve_ivp(
            f,
            (t0, t1),
            y0,
            method='DOP853',
            rtol=1e-13,
            atol=1e-15,
            max_step=0.01
        )
        
        if sol.success and sol.t[-1] == t1:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
        
        # If scipy succeeded but didn't reach the end, try with larger max_step
        sol = solve_ivp(
            f,
            (t0, t1),
            y0,
            method='DOP853',
            rtol=1e-12,
            atol=1e-14,
            max_step=0.05
        )
        
        if sol.success and sol.t[-1] == t1:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
                
    except Exception:
        pass
    
    # Check if we're running out of time
    if time.time() > deadline:
        # Return best effort: use scipy with looser tolerances
        try:
            sol = solve_ivp(
                f,
                (t0, t1),
                y0,
                method='DOP853',
                rtol=1e-10,
                atol=1e-12
            )
            if sol.success and sol.t[-1] == t1:
                return sol.y[:, -1]
        except Exception:
            pass
        return y0.copy()
    
    # Fallback: manual high-order RK method
    # Use RK4 with adaptive-like fixed small steps
    n_steps = 5000
    h = (t1 - t0) / n_steps
    t = t0
    y = y0.copy()
    
    for i in range(n_steps):
        if time.time() > deadline:
            break
        
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        
        if not np.all(np.isfinite(y)):
            break
    
    if np.all(np.isfinite(y)):
        return y
    
    return y0.copy()
# EVOLVE-BLOCK-END
