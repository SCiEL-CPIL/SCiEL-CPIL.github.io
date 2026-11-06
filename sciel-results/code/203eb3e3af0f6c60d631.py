"""ODE solver: Convection-diffusion 100-point using scipy adaptive stiff solvers."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    deadline = time.time() + 280.0  # safety margin before 330s kill

    # Primary solve: BDF (implicit, stiff) with adaptive stepping
    # rtol=1e-12 balances accuracy vs. step count (measured: prevents stagnation)
    try:
        sol = solve_ivp(
            f, (t0, t1), y0_arr,
            method='BDF',
            rtol=1e-12, atol=1e-14,
            max_step=(t1 - t0) / 50.0,  # cap step size for accuracy
        )
        if sol.success and sol.y.shape[1] > 0:
            y_final = sol.y[:, -1]
            if np.all(np.isfinite(y_final)):
                # Check if we have time for a polish pass
                if time.time() < deadline - 30.0:
                    # Polish: re-solve from t0 with tighter tolerance using the
                    # primary solution as initial guess at intermediate points
                    # Actually, just do a second solve with even tighter rtol
                    try:
                        sol2 = solve_ivp(
                            f, (t0, t1), y0_arr,
                            method='BDF',
                            rtol=1e-13, atol=1e-15,
                            max_step=(t1 - t0) / 50.0,
                        )
                        if sol2.success and sol2.y.shape[1] > 0:
                            y_final = sol2.y[:, -1]
                            if np.all(np.isfinite(y_final)):
                                return y_final
                    except Exception:
                        pass
                return y_final
    except Exception:
        pass

    # Fallback: LSODA (auto-switches between stiff/non-stiff)
    try:
        sol = solve_ivp(
            f, (t0, t1), y0_arr,
            method='LSODA',
            rtol=1e-12, atol=1e-14,
            max_step=(t1 - t0) / 50.0,
        )
        if sol.success and sol.y.shape[1] > 0:
            y_final = sol.y[:, -1]
            if np.all(np.isfinite(y_final)):
                return y_final
    except Exception:
        pass

    # Last resort: fixed-step RK4 (same as seed)
    steps = 2000
    h = (t1 - t0) / steps
    t, y = t0, y0_arr.copy()
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            break
    return y
# EVOLVE-BLOCK-END
