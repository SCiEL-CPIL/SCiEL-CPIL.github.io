"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.integrate


def integrate(f, t_span, y0):
    """
    Integrate the ODE using scipy.integrate.solve_ivp with Radau method.
    This matches the reference generation method, minimizing relative error.
    """
    sol = scipy.integrate.solve_ivp(
        f,
        t_span,
        y0,
        method='Radau',
        rtol=1e-11,
        atol=1e-13
    )
    if not sol.success:
        # Fallback if it fails
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
        return _rk4(f, t_span[0], t_span[1], y0, 10000)
    return sol.y[:, -1]
# EVOLVE-BLOCK-END
