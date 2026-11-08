import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 285  # 15s margin before 300s limit
    
    y0 = np.asarray(y0, dtype=float)
    
    # Strategy: Use DOP853 (8th order explicit) with very tight tolerances
    # For the 2D heat equation (stiff but smooth), high-order explicit methods
    # with tight tolerances can achieve excellent accuracy with moderate step counts.
    # Alternative: Radau IIA (implicit) could reduce calls but may be slower per step
    # for this problem size. Let's try DOP853 first with the tightest practical tolerances.
    
    try:
        # First attempt: DOP853 with extremely tight tolerances
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-15,
            atol=1e-15,
            max_step=0.005,  # Limit step size for accuracy control
            first_step=1e-5  # Start small to establish accuracy
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            # If DOP853 failed, try with slightly relaxed tolerances
            sol = solve_ivp(
                f,
                t_span,
                y0,
                method='DOP853',
                rtol=1e-13,
                atol=1e-13,
                max_step=0.005
            )
            if sol.success:
                return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback: Radau IIA (implicit, good for stiff problems)
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-13,
            atol=1e-13,
            max_step=0.01
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Last resort: RK45 with moderate tolerances
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='RK45',
            rtol=1e-12,
            atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Final fallback: manual RK4
    def _rk4(f, t0, t1, y0, steps):
        h = (t1 - t0) / steps
        t, y = t0, np.array(y0, dtype=float)
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        return y
    
    return _rk4(f, t_span[0], t_span[1], y0, 5000)
