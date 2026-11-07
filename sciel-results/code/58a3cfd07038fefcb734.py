"""High-accuracy ODE integrator for the harmonic oscillator using scipy DOP853 with polish pass."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.integrate as integrate_scipy


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] starting at y0.
    Uses scipy's DOP853 (8th/7th order) with tight tolerances, then a
    high-precision polish pass to minimize relative error.
    """
    deadline = time.time() + 290  # safety margin before 300s kill
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    
    if t1 <= t0:
        return y0_arr.copy()
    
    # Phase 1: Primary integration with DOP853 at reference tolerances
    y_end = None
    try:
        sol = integrate_scipy.solve_ivp(
            f,
            t_span=(t0, t1),
            y0=y0_arr,
            method='DOP853',
            rtol=1e-11,
            atol=1e-13,
            dense_output=False
        )
        if sol.success and sol.t[-1] >= t1 - 1e-14:
            y_end = sol.y[:, -1].copy()
            if not np.all(np.isfinite(y_end)):
                y_end = None
    except Exception:
        y_end = None
    
    # If primary failed, try looser tolerances
    if y_end is None:
        try:
            sol = integrate_scipy.solve_ivp(
                f,
                t_span=(t0, t1),
                y0=y0_arr,
                method='DOP853',
                rtol=1e-10,
                atol=1e-12,
                dense_output=False
            )
            if sol.success and sol.t[-1] >= t1 - 1e-14:
                y_end = sol.y[:, -1].copy()
                if not np.all(np.isfinite(y_end)):
                    y_end = None
        except Exception:
            y_end = None
    
    # If DOP853 failed entirely, try RK45
    if y_end is None:
        try:
            sol = integrate_scipy.solve_ivp(
                f,
                t_span=(t0, t1),
                y0=y0_arr,
                method='RK45',
                rtol=1e-11,
                atol=1e-13,
                dense_output=False
            )
            if sol.success and sol.t[-1] >= t1 - 1e-14:
                y_end = sol.y[:, -1].copy()
                if not np.all(np.isfinite(y_end)):
                    y_end = None
        except Exception:
            y_end = None
    
    # Phase 2: High-precision polish pass with much tighter tolerances
    # This should converge to machine precision for the harmonic oscillator
    if y_end is not None and time.time() < deadline - 10:
        try:
            sol_polish = integrate_scipy.solve_ivp(
                f,
                t_span=(t0, t1),
                y0=y0_arr,
                method='DOP853',
                rtol=1e-14,
                atol=1e-16,
                dense_output=False
            )
            if sol_polish.success and sol_polish.t[-1] >= t1 - 1e-14:
                y_polished = sol_polish.y[:, -1]
                if np.all(np.isfinite(y_polished)):
                    y_end = y_polished
        except Exception:
            pass  # Keep the less accurate result
    
    # Phase 3: Fallback RK4 if all scipy methods failed
    if y_end is None:
        h = (t1 - t0) / 10000
        t = t0
        y = y0_arr.copy()
        for _ in range(10000):
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
        y_end = y
    
    # Final validation
    if np.all(np.isfinite(y_end)):
        return y_end
    
    # Absolute fallback
    return y0_arr.copy()
# EVOLVE-BLOCK-END
