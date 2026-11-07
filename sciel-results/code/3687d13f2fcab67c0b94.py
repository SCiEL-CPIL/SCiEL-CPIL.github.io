"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator (mu=5) from t_span[0] to t_span[1]
    with initial condition y0, returning y at t_span[1].
    
    The problem is fixed: y0' = y1, y1' = 5*(1-y0^2)*y1 - y0, t in [0,10].
    We use a single high-accuracy solve_ivp call with Radau method.
    """
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Use Radau with very tight tolerances for near-machine precision
    sol = solve_ivp(
        f,
        t_span=(t0, t1),
        y0=y0_arr,
        method='Radau',
        rtol=1e-13,
        atol=1e-15,
        dense_output=False
    )
    
    if not sol.success:
        # Fallback: try with slightly looser tolerances
        sol = solve_ivp(
            f,
            t_span=(t0, t1),
            y0=y0_arr,
            method='Radau',
            rtol=1e-11,
            atol=1e-13,
            dense_output=False
        )
    
    return sol.y[:, -1]
# EVOLVE-BLOCK-END
