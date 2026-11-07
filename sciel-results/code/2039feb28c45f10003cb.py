"""ODE integrator for the Van der Pol oscillator (mu = 5)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator from t_span[0] to t_span[1]
    with initial condition y0, using a high-accuracy implicit method.
    
    Returns the state at t_span[1].
    """
    t_start = time.time()
    deadline = t_start + 280.0  # safety margin before 300s limit
    
    y0 = np.asarray(y0, dtype=float)
    t0, t1 = t_span
    
    # Use Radau with very tight tolerances to drive error to the floor.
    # The reference uses rtol=1e-11, atol=1e-13.
    # We use rtol=1e-14, atol=1e-16 to get as close to the reference as possible.
    
    # First, try with very tight tolerances
    try:
        sol = solve_ivp(
            f, (t0, t1), y0,
            method='Radau',
            rtol=1e-14,
            atol=1e-16,
            max_step=0.1
        )
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback: slightly looser tolerances
    try:
        sol = solve_ivp(
            f, (t0, t1), y0,
            method='Radau',
            rtol=1e-13,
            atol=1e-15,
            max_step=0.1
        )
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Final fallback: even looser
    try:
        sol = solve_ivp(
            f, (t0, t1), y0,
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
            max_step=0.1
        )
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Last resort: RK45 with moderate tolerances
    try:
        sol = solve_ivp(
            f, (t0, t1), y0,
            method='RK45',
            rtol=1e-10,
            atol=1e-12,
            max_step=0.01
        )
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
    except Exception:
        pass
    
    # If all else fails, return y0 as a last resort (should not happen)
    return y0.copy()
# EVOLVE-BLOCK-END
