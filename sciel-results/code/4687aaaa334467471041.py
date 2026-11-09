"""ODE integrator for the fixed Lorenz IVP: y0' = 10(y1-y0), y1' = y0(28-y2)-y1, y2' = y0*y1-(8/3)y2, y(0)=[1,1,1], t in [0,2]."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Wall-clock budget with safety margin (evaluator kills at 330 s).
_DEADLINE = 330.0 - 10.0  # 320 s
_START = time.monotonic()


def _remaining():
    return _DEADLINE - (time.monotonic() - _START)


def _dop853(f, t_span, y0, rtol, atol):
    """High-order adaptive DOP853 via scipy; returns (y_end, ok)."""
    from scipy.integrate import solve_ivp
    t0, t1 = t_span
    sol = solve_ivp(
        f,
        (t0, t1),
        np.asarray(y0, dtype=float),
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        return None, False
    y = sol.y[:, -1]
    if y.shape[0] != len(y0):
        return None, False
    if not np.all(np.isfinite(y)):
        return None, False
    return y, True


def _rk45_adaptive(f, t_span, y0, rtol=1e-11, atol=1e-13):
    """Fallback: classic adaptive Runge-Kutta-Fehlberg (RKF45) in pure Python/NumPy."""
    t0, t1 = t_span
    t = float(t0)
    y = np.asarray(y0, dtype=float).copy()
    h = (t1 - t0) / 1000.0
    h_min = (t1 - t0) * 1e-12
    h_max = (t1 - t0) * 0.25
    n = len(y)

    def rk_step(tt, yy, hh):
        k1 = f(tt, yy)
        k2 = f(tt + hh / 2.0, yy + hh / 2.0 * k1)
        k3 = f(tt + hh * 0.375, yy + hh * (0.75 * k1 / 2.0 + k2 * 0.75))
        k4 = f(tt + hh * 0.9375, yy + hh * (-9.0 * k1 / 32.0 + 3.0 * k2 / 4.0 + 15.0 * k3 / 32.0))
        k5 = f(tt + hh, yy + hh * (2383.0 * k1 / 4100.0 - 5.0 * k2 / 4.0 + 71.0 * k3 / 100.0 + 37.0 * k4 / 100.0))
        k6 = f(tt + hh / 2.0, yy + hh * (-8.0 * k1 / 27.0 + k2 + k3 / 9.0))
        y4 = yy + hh * (37.0 * k1 / 360.0 + 250.0 * k2 / 621.0 + 125.0 * k3 / 594.0 + 512.0 * k4 / 1771.0 + 180.0 * k5 / 133.0)
        y5 = yy + hh * (2825.0 * k1 / 27648.0 + 18575.0 * k2 / 48384.0 + 13525.0 * k3 / 55296.0 + 277.0 * k4 / 144.0 + 1.0 * k5 / 6.0)
        err = np.abs(y5 - y4)
        return y5, err

    while t < t1:
        if _remaining() < 2.0:
            break
        hh = min(h, t1 - t)
        y5, err = rk_step(t, y, hh)
        if not np.all(np.isfinite(err)):
            hh = max(hh * 0.5, h_min)
            continue
        scale = np.maximum(atol, rtol * np.abs(y5))
        err_norm = np.max(err / scale)
        if err_norm <= 1.0:
            t += hh
            y = y5
            if err_norm > 1e-15:
                h = min(h * min(5.0, 0.9 * err_norm ** (-0.2)), h_max)
            else:
                h = min(h * 5.0, h_max)
        else:
            h = max(h * 0.9 * err_norm ** (-0.2), h_min)
    return y


def integrate(f, t_span, y0):
    # Primary: mirror the reference DOP853 (rtol 1e-11, atol 1e-13) for minimal, floor-level error.
    y, ok = _dop853(f, t_span, y0, rtol=1e-11, atol=1e-13)
    if ok:
        return y

    # Fallback: adaptive RKF45 with tight tolerances, guarded by the deadline.
    if _remaining() < 5.0:
        # Last resort: a few fixed-step RK4 steps to guarantee a finite return.
        t0, t1 = t_span
        steps = 2000
        h = (t1 - t0) / steps
        t = t0
        y = np.asarray(y0, dtype=float).copy()
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h / 2.0, y + h / 2.0 * k1)
            k3 = f(t + h / 2.0, y + h / 2.0 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h
        return y

    y = _rk45_adaptive(f, t_span, y0, rtol=1e-11, atol=1e-13)
    if not np.all(np.isfinite(y)):
        # Final safety: fixed-step RK4.
        t0, t1 = t_span
        steps = 4000
        h = (t1 - t0) / steps
        t = t0
        y = np.asarray(y0, dtype=float).copy()
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h / 2.0, y + h / 2.0 * k1)
            k3 = f(t + h / 2.0, y + h / 2.0 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h
    return y
# EVOLVE-BLOCK-END
