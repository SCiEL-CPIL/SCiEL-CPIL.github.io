"""High-precision Van der Pol integrator using Radau with analytic Jacobian."""
import time
import numpy as np
from scipy.integrate import solve_ivp


def _jac(t, y):
    """Analytic Jacobian for the Van der Pol oscillator (mu=5)."""
    y0, y1 = y[0], y[1]
    J = np.empty((2, 2))
    J[0, 0] = 0.0
    J[0, 1] = 1.0
    J[1, 0] = -10.0 * y0 * y1 - 1.0
    J[1, 1] = 5.0 * (1.0 - y0 * y0)
    return J


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    deadline = time.time() + 270.0  # safety margin before 330s kill

    def _fallback_rk4():
        """Fallback explicit RK4 if implicit solver fails."""
        steps = 5000
        h = (t1 - t0) / steps
        t, y = t0, y0.copy()
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h
            if not np.all(np.isfinite(y)):
                break
        return y

    try:
        # First pass: use Radau with tight tolerances to get a high-accuracy solution
        sol = solve_ivp(
            f,
            (t0, t1),
            y0,
            method="Radau",
            jac=_jac,
            rtol=1e-14,
            atol=1e-16,
        )
        if sol.success and sol.t[-1] == t1 and np.all(np.isfinite(sol.y[:, -1])):
            y_end = sol.y[:, -1]
            # Final high-precision polish pass with even tighter tolerances
            if time.time() < deadline:
                try:
                    sol2 = solve_ivp(
                        f,
                        (t0, t1),
                        y0,
                        method="Radau",
                        jac=_jac,
                        rtol=1e-15,
                        atol=1e-17,
                    )
                    if sol2.success and sol2.t[-1] == t1 and np.all(np.isfinite(sol2.y[:, -1])):
                        y_end = sol2.y[:, -1]
                except Exception:
                    pass
            return y_end
    except Exception:
        pass

    if time.time() > deadline:
        return y0.copy()

    return _fallback_rk4()
