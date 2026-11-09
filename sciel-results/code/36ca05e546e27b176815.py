"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses solve_ivp with Radau (L-stable) for the stiff Robertson problem,
    followed by a high-precision polish pass if time permits.
    Returns the final state y(t_end).
    """
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    
    # Track elapsed time for deadline guard
    start_time = time.time()
    deadline = start_time + 280  # Leave 50s margin for 330s limit
    
    def time_left():
        return deadline - time.time()
    
    # Primary: Radau with tight tolerances
    try:
        sol = solve_ivp(
            f, (t0, t1), y0_arr,
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
        )
        if sol.success and sol.y.shape[1] > 0:
            y_final = sol.y[:, -1].copy()
        else:
            raise ValueError("solve_ivp Radau failed")
    except Exception:
        # Fallback: LSODA via odeint
        try:
            from scipy.integrate import odeint
            def f_odeint(y, t):
                return f(t, y)
            sol = odeint(f_odeint, y0_arr, [t1], method='LSODA',
                         rtol=1e-12, atol=1e-14)
            y_final = sol[-1].copy()
        except Exception:
            # Last resort: BDF
            try:
                sol = solve_ivp(
                    f, (t0, t1), y0_arr,
                    method='BDF',
                    rtol=1e-12,
                    atol=1e-14,
                )
                if sol.success:
                    y_final = sol.y[:, -1].copy()
                else:
                    raise ValueError("BDF failed")
            except Exception:
                # RK4 fallback
                h = (t1 - t0) / 20000
                t, y = t0, y0_arr.copy()
                n_steps = int((t1 - t0) / h)
                for i in range(n_steps):
                    if time_left() < 5:
                        break
                    k1 = f(t, y)
                    k2 = f(t + h/2, y + h/2 * k1)
                    k3 = f(t + h/2, y + h/2 * k2)
                    k4 = f(t + h, y + h * k3)
                    y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                    t += h
                y_final = y
    
    # High-precision polish pass if time permits
    if time_left() > 30:
        try:
            sol2 = solve_ivp(
                f, (t0, t1), y0_arr,
                method='Radau',
                rtol=1e-14,
                atol=1e-16,
            )
            if sol2.success and sol2.y.shape[1] > 0:
                y_final = sol2.y[:, -1]
        except Exception:
            pass
    
    return y_final
# EVOLVE-BLOCK-END
