"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns the final state y(t_end).
    
    Uses LSODA method from scipy.integrate.solve_ivp which automatically
    switches between Adams (non-stiff) and BDF (stiff) formulas.
    Tolerances are set to balance high accuracy with computational efficiency.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Use LSODA with high accuracy tolerances but allowing adaptive step sizing
    # without artificial max_step restrictions to minimize function calls.
    sol = solve_ivp(
        f,
        t_span,
        y0,
        method='LSODA',
        rtol=1e-10,
        atol=1e-12,
    )
    
    if not sol.success:
        # Fallback: if LSODA fails, try Radau which is robust for stiff problems
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-10,
            atol=1e-12,
        )
    
    return sol.y[:, -1]
# EVOLVE-BLOCK-END
