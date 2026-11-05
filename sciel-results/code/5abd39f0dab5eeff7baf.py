"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _robertson_jac(t, y):
    """Analytical Jacobian for the Robertson problem."""
    y0, y1, y2 = y
    return np.array([
        [-0.04, 1e4 * y2, 1e4 * y1],
        [0.04, -1e4 * y2 - 3e7 * y1, -1e4 * y1],
        [0.0, 3e7 * y1, 0.0]
    ])


def integrate(f, t_span, y0):
    """
    Solve the Robertson ODE problem y' = f(t, y) on [t0, t1].
    Returns the final state y(t1).
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard: 300s limit, keep a safety margin.
    start_time = time.time()
    deadline = start_time + 285  # 15s safety margin
    
    from scipy.integrate import solve_ivp
    
    # Try Radau first with very tight tolerances and analytical Jacobian
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            jac=_robertson_jac,
            rtol=1e-14,
            atol=1e-16,
            max_step=10.0
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # If Radau fails, try BDF with Jacobian
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='BDF',
            jac=_robertson_jac,
            rtol=1e-14,
            atol=1e-16
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # If both fail, try Radau without Jacobian
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-13,
            atol=1e-15
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback to BDF without Jacobian
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='BDF',
            rtol=1e-13,
            atol=1e-15
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Last resort: simple RK4 with small steps
    h = (t1 - t0) / 20000
    t = t0
    y = np.array(y0, dtype=float)
    while t < t1:
        if time.time() > deadline:
            break
        h_step = min(h, t1 - t)
        k1 = f(t, y)
        k2 = f(t + h_step/2, y + h_step/2 * k1)
        k3 = f(t + h_step/2, y + h_step/2 * k2)
        k4 = f(t + h_step, y + h_step * k3)
        y = y + h_step/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h_step
    return y

# EVOLVE-BLOCK-END
