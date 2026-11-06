"""Rewritten ODE solver for the Robertson problem using LSODA with adaptive strategy."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns the final state y(t_end).
    
    Uses LSODA with a two-phase strategy:
    - Phase 1: Tight tolerances on the stiff initial transient [t0, min(t0+2, t1)]
    - Phase 2: Relaxed tolerances on the slow decay phase [phase1_end, t1]
    This reduces function evaluations in the non-stiff tail while maintaining
    high accuracy through the stiff region.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    total_span = t1 - t0
    
    # Phase 1: Stiff initial transient
    # The Robertson problem has a fast transient in the first ~1-2 time units
    phase1_end = min(t0 + 2.0, t1)
    
    if phase1_end <= t0:
        # Degenerate case, just solve the whole interval
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='LSODA',
            rtol=1e-13,
            atol=1e-15,
        )
        return sol.y[:, -1]
    
    # Solve phase 1 with very tight tolerances
    sol1 = solve_ivp(
        f,
        [t0, phase1_end],
        y0,
        method='LSODA',
        rtol=1e-13,
        atol=1e-15,
    )
    
    if not sol1.success:
        # Fallback: solve the whole interval with moderate tolerances
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='LSODA',
            rtol=1e-12,
            atol=1e-14,
        )
        return sol.y[:, -1]
    
    y_phase1_end = sol1.y[:, -1]
    
    # Phase 2: Slow decay
    if phase1_end >= t1:
        return y_phase1_end
    
    sol2 = solve_ivp(
        f,
        [phase1_end, t1],
        y_phase1_end,
        method='LSODA',
        rtol=1e-11,
        atol=1e-13,
    )
    
    if not sol2.success:
        # Fallback: re-solve phase 2 with tighter tolerances
        sol2 = solve_ivp(
            f,
            [phase1_end, t1],
            y_phase1_end,
            method='LSODA',
            rtol=1e-13,
            atol=1e-15,
        )
        if not sol2.success:
            # Last resort: solve entire interval with Radau
            sol = solve_ivp(
                f,
                t_span,
                y0,
                method='Radau',
                rtol=1e-12,
                atol=1e-14,
            )
            return sol.y[:, -1]
    
    return sol2.y[:, -1]
# EVOLVE-BLOCK-END
