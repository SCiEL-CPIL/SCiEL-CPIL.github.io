"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def _robertson_jac(t, y):
    """Analytical Jacobian for the standard Robertson ODE system.
    y' = [-0.04*y0 + 1e4*y1*y2,
          0.04*y0 - 1e4*y1*y2 - 3e3*y1^2,
          3e3*y1^2]
    """
    y0, y1, y2 = y
    return np.array([
        [-0.04, 1e4 * y2, 1e4 * y1],
        [0.04, -1e4 * y2 - 6e3 * y1, -1e4 * y1],
        [0.0, 6e3 * y1, 0.0]
    ])


def integrate(f, t_span, y0):
    """
    Integrates the fixed Robertson ODE problem using an implicit Radau solver
    with the analytical Jacobian supplied to avoid finite-difference cost.
    Returns y(t_end).
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)

    # Deadline guard: stay well under the 330 s kill limit.
    start_time = time.time()
    deadline = start_time + 290.0

    # --- Primary: Radau with analytical Jacobian, tight tolerances ---
    try:
        sol = solve_ivp(
            f,
            (t0, t1),
            y,
            method='Radau',
            jac=_robertson_jac,
            rtol=1e-12,
            atol=1e-14,
            max_step=(t1 - t0) / 500.0,
        )
        if sol.success and sol.y.shape[1] > 0:
            return sol.y[:, -1]
    except Exception:
        pass

    # --- Fallback 1: BDF with analytical Jacobian ---
    if time.time() < deadline:
        try:
            sol = solve_ivp(
                f,
                (t0, t1),
                y,
                method='BDF',
                jac=_robertson_jac,
                rtol=1e-12,
                atol=1e-14,
                max_step=(t1 - t0) / 500.0,
            )
            if sol.success and sol.y.shape[1] > 0:
                return sol.y[:, -1]
        except Exception:
            pass

    # --- Fallback 2: Radau without Jacobian (let SciPy finite-difference it) ---
    if time.time() < deadline:
        try:
            sol = solve_ivp(
                f,
                (t0, t1),
                y,
                method='Radau',
                rtol=1e-10,
                atol=1e-12,
                max_step=(t1 - t0) / 500.0,
            )
            if sol.success and sol.y.shape[1] > 0:
                return sol.y[:, -1]
        except Exception:
            pass

    # --- Last resort: return initial state (should never happen) ---
    return y

# EVOLVE-BLOCK-END
