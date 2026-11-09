"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses adaptive RK45 with tight tolerances for the Robertson problem.
    Returns the final state y(t_end).
    """
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    
    # Use solve_ivp with RK45 and very tight tolerances
    # Robertson problem is stiff, so we might need BDF or Radau, but RK45 with small steps can work
    # Actually, for stiff problems like Robertson, implicit methods are better.
    # Let's try Radau first as it's good for stiff problems.
    
    deadline = time.time() + 290  # Leave 10s margin for 300s limit
    
    try:
        # Try Radau method with tight tolerances
        sol = solve_ivp(
            f, (t0, t1), y0_arr,
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
            max_step=0.1
        )
        if sol.success and time.time() < deadline:
            return sol.y[:, -1]
    except Exception:
        pass
    
    try:
        # Try BDF method
        sol = solve_ivp(
            f, (t0, t1), y0_arr,
            method='BDF',
            rtol=1e-12,
            atol=1e-14,
            max_step=0.1
        )
        if sol.success and time.time() < deadline:
            return sol.y[:, -1]
    except Exception:
        pass
    
    try:
        # Try LSODA
        sol = solve_ivp(
            f, (t0, t1), y0_arr,
            method='LSODA',
            rtol=1e-12,
            atol=1e-14,
            max_step=0.1
        )
        if sol.success and time.time() < deadline:
            return sol.y[:, -1]
    except Exception:
        pass
    
    try:
        # Fallback to RK45 with very small max step
        sol = solve_ivp(
            f, (t0, t1), y0_arr,
            method='RK45',
            rtol=1e-10,
            atol=1e-12,
            max_step=0.01
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Last resort: simple RK4 with many steps
    h = (t1 - t0) / 100000
    t, y = t0, y0_arr.copy()
    n_steps = 100000
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
    
    return y
# EVOLVE-BLOCK-END
