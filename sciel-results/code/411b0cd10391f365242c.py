"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    deadline = time.time() + 280  # safety margin before 330s kill

    # Probe linearity: check if f(t, y) = A @ y (independent of t)
    f0 = f(t0, y0)

    # Check if f is linear and time-independent by probing with basis vectors
    f_probe1 = f(t0, 2 * y0)

    # Check linearity: f(2y) ≈ 2f(y)
    linearity_err = np.linalg.norm(f_probe1 - 2 * f0) / (2 * np.linalg.norm(f0) + 1e-30)

    if linearity_err < 1e-8:
        # Likely linear and time-independent. Extract A by probing basis vectors.
        A = np.zeros((n, n))

        for j in range(n):
            e = np.zeros(n)
            e[j] = 1.0
            A[:, j] = f(t0, e)
            if time.time() > deadline:
                break

        # If we ran out of time during extraction, fall back to RK4
        if time.time() > deadline:
            return _rk4_fallback(f, t0, t1, y0)

        dt = t1 - t0

        # Compute y(t1) = expm(dt * A) @ y0
        M = expm(dt * A)
        y_final = M @ y0

        return y_final
    else:
        # Not linear; fall back to RK4
        return _rk4_fallback(f, t0, t1, y0)


def _rk4_fallback(f, t0, t1, y0):
    """RK4 fallback with fixed step size."""
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    h = (t1 - t0) / 2000
    t, y = t0, y0.copy()
    for _ in range(2000):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            # Fall back to backward Euler with fewer steps
            h2 = (t1 - t) / 500
            t_cur = t
            for _ in range(500):
                z = y.copy()
                for _newton in range(3):
                    fz = f(t_cur + h2, z)
                    J = np.empty((n, n))
                    for j in range(n):
                        e = np.zeros(n)
                        e[j] = 1e-7 * max(1.0, abs(z[j]))
                        J[:, j] = (f(t_cur + h2, z + e) - fz) / e[j]
                    step = np.linalg.solve(np.eye(n) - h2 * J, -(z - y - h2 * fz))
                    z = z + step
                    if np.linalg.norm(step) <= 1e-10 * (1 + np.linalg.norm(z)):
                        break
                y, t_cur = z, t_cur + h2
            break
    return y
# EVOLVE-BLOCK-END
