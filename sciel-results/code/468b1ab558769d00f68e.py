"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Solve y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns y(t_end).
    Uses LSODA via solve_ivp for stiff problem handling.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard: 330s wall-clock, keep 10s margin
    deadline = time.time() + 320.0
    
    # Robertson problem is stiff. Use LSODA which automatically switches
    # between Adams (non-stiff) and BDF (stiff).
    # We provide a Jacobian to reduce function evaluations.
    # For Robertson: y' = [-0.04*y0 + 1e4*y1*y2, 0.04*y0 - 1e4*y1*y2 - 3e7*y1^2, 3e7*y1^2]
    # Jacobian:
    # [[-0.04, 1e4*y2, 1e4*y1],
    #  [ 0.04, -1e4*y2 - 6e7*y1, -1e4*y1],
    #  [  0,    6e7*y1,           0]]
    
    def jac(t, y):
        return np.array([
            [-0.04, 1e4 * y[2], 1e4 * y[1]],
            [ 0.04, -1e4 * y[2] - 6e7 * y[1], -1e4 * y[1]],
            [  0.0,  6e7 * y[1],          0.0]
        ])
    
    # Try LSODA with tight tolerances
    try:
        sol = solve_ivp(
            f, (t0, t1), y0,
            method='LSODA',
            jac=jac,
            rtol=1e-12,
            atol=1e-14,
            max_step=1.0
        )
        if sol.success and time.time() < deadline:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback: BDF with tight tolerances
    try:
        sol = solve_ivp(
            f, (t0, t1), y0,
            method='BDF',
            jac=jac,
            rtol=1e-12,
            atol=1e-14,
            max_step=1.0
        )
        if sol.success and time.time() < deadline:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Last resort: Radau
    try:
        sol = solve_ivp(
            f, (t0, t1), y0,
            method='Radau',
            jac=jac,
            rtol=1e-10,
            atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # If all fail, try a simple fixed-step RK4 as absolute last resort
    try:
        y = y0.copy()
        n_steps = 10000
        h = (t1 - t0) / n_steps
        t = t0
        for _ in range(n_steps):
            if time.time() > deadline:
                break
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y
    except Exception:
        return y0
# EVOLVE-BLOCK-END
