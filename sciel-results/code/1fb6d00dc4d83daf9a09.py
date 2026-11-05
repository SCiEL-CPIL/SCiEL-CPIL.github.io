def _rk4(f, t0, t1, y0, steps, deadline=None):
    """Fixed-step RK4 integrator with basic safety checks."""
    h = (t1 - t0) / steps
    t = float(t0)
    y = np.asarray(y0, dtype=float).copy()
    n = len(y)
    for i in range(steps):
        if deadline is not None and time.time() > deadline:
            return y
        k1 = np.asarray(f(t, y), dtype=float)
        k2 = np.asarray(f(t + 0.5 * h, y + 0.5 * h * k1), dtype=float)
        k3 = np.asarray(f(t + 0.5 * h, y + 0.5 * h * k2), dtype=float)
        k4 = np.asarray(f(t + h, y + h * k3), dtype=float)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            return None
    return y
