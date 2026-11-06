"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def _robertson_jac(t, y):
    """Analytical Jacobian for the Robertson problem."""
    y0, y1, y2 = y
    k1, k2, k3 = 0.04, 3.0e-5, 1.0e4
    return np.array([
        [-k1, 0.0, k3],
        [k1, -(k1 + k2), 0.0],
        [0.0, k2, -k2]
    ])


def integrate(f, t_span, y0):
    """
    Integrate the Robertson ODE problem on [0, 40] using a hybrid strategy:
    1. Radau (implicit) for the initial transient [0, t_switch] where the problem is stiff.
    2. DOP853 (explicit) for the remaining interval [t_switch, t_end] where the problem
       is effectively non-stiff and explicit high-order methods are more efficient.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0

    # Switch point: Fast modes decay very quickly.
    t_switch = 1.0
    
    if t_switch > t1:
        t_switch = t1

    try:
        # Phase 1: Stiff region [t0, t_switch]
        sol1 = solve_ivp(
            f,
            [t0, t_switch],
            y0,
            method='Radau',
            jac=_robertson_jac,
            rtol=1e-12,
            atol=1e-14,
        )
        
        if not sol1.success:
            # Fallback to full Radau if phase 1 fails
            sol_full = solve_ivp(
                f,
                t_span,
                y0,
                method='Radau',
                jac=_robertson_jac,
                rtol=1e-12,
                atol=1e-14,
            )
            return sol_full.y[:, -1] if sol_full.success else y0

        y_switch = sol1.y[:, -1]
        
        if time.time() > deadline:
            return y_switch

        # Phase 2: Non-stiff region [t_switch, t1]
        sol2 = solve_ivp(
            f,
            [t_switch, t1],
            y_switch,
            method='DOP853',
            rtol=1e-12,
            atol=1e-14,
        )
        
        if sol2.success:
            return sol2.y[:, -1]
        else:
            # Fallback to Radau for the rest if DOP853 fails
            sol2_fallback = solve_ivp(
                f,
                [t_switch, t1],
                y_switch,
                method='Radau',
                jac=_robertson_jac,
                rtol=1e-12,
                atol=1e-14,
            )
            if sol2_fallback.success:
                return sol2_fallback.y[:, -1]
            return y_switch

    except Exception:
        # Final fallback: Full Radau
        try:
            sol = solve_ivp(
                f,
                t_span,
                y0,
                method='Radau',
                jac=_robertson_jac,
                rtol=1e-12,
                atol=1e-14,
            )
            if sol.success:
                return sol.y[:, -1]
            return sol.y[:, -1] if sol.y.size > 0 else y0
        except Exception:
            return y0


# EVOLVE-BLOCK-END
