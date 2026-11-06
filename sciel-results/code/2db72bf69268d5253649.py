"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.integrate


def _dop853(f, t0, t1, y0, rtol=1e-10, atol=1e-12, deadline=None):
    """High-order explicit Runge-Kutta (DOP853) with adaptive stepping."""
    y0 = np.asarray(y0, dtype=float)
    sol = scipy.integrate.solve_ivp(
        f, (t0, t1), y0,
        method='DOP853',
        rtol=rtol,
        atol=atol,
        max_step=np.inf
    )
    if sol.success:
        return sol.y[:, -1]
    return None


def _backward_euler(f, t0, t1, y0, steps):
    # Implicit Euler with a finite-difference Jacobian and a few Newton iterations (handles stiff problems).
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    n = len(y)
    for _ in range(steps):
        z = y.copy()
        for _newton in range(4):
            fz = f(t + h, z)
            J = np.empty((n, n))
            for j in range(n):
                e = np.zeros(n)
                e[j] = 1e-7 * max(1.0, abs(z[j]))
                J[:, j] = (f(t + h, z + e) - fz) / e[j]
            step = np.linalg.solve(np.eye(n) - h * J, -(z - y - h * fz))
            z = z + step
            if np.linalg.norm(step) <= 1e-10 * (1 + np.linalg.norm(z)):
                break
        y, t = z, t + h
    return y


def integrate(f, t_span, y0):
    # Try high-order explicit method first; fall back to implicit for stiff problems.
    t0, t1 = t_span
    start_time = time.time()
    
    # DOP853 with tight tolerances
    y = _dop853(f, t0, t1, y0, rtol=1e-10, atol=1e-12)
    if y is not None:
        return y
    
    # Fallback to implicit Euler if DOP853 fails
    y = _backward_euler(f, t0, t1, y0, 2000)
    return y
# EVOLVE-BLOCK-END
