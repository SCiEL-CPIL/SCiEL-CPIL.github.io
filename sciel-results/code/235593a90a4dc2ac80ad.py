"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin from 330s limit
    
    try:
        from scipy.integrate import solve_ivp
    except ImportError:
        # Fallback to manual RK45 if scipy not available
        def _rk45(f, t0, t1, y0, rtol=1e-10, atol=1e-12):
            y = np.array(y0, dtype=float)
            t = t0
            h = (t1 - t0) / 100.0
            while t < t1 and time.time() < deadline:
                h = min(h, t1 - t)
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2*k1)
                k3 = f(t + h/2, y + h/2*k2)
                k4 = f(t + h, y + h*k3)
                y_new = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
                y = y_new
                t += h
                h *= 1.1
            return y
        return _rk45(f, t0, t1, y0)

    # Strategy: Use Radau with progressively tighter tolerances to maximize accuracy
    # while keeping function calls manageable. The Robertson problem is stiff,
    # so implicit methods like Radau are appropriate.
    
    # First pass: moderate tolerances to get a good solution quickly
    try:
        sol = solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=1e-10, atol=1e-12
        )
        if sol.success:
            y_current = sol.y[:, -1].copy()
        else:
            y_current = np.array(y0, dtype=float)
    except Exception:
        y_current = np.array(y0, dtype=float)
    
    # Check if we have time for a second, more accurate pass
    elapsed = time.time() - start_time
    if elapsed < deadline - 30.0:
        # Second pass: tighter tolerances for higher accuracy
        try:
            sol2 = solve_ivp(
                f, t_span, y0,
                method='Radau',
                rtol=1e-12, atol=1e-14
            )
            if sol2.success:
                y_current = sol2.y[:, -1]
        except Exception:
            pass
    
    # Check if we have time for a third, even more accurate pass
    elapsed = time.time() - start_time
    if elapsed < deadline - 10.0:
        # Third pass: very tight tolerances
        try:
            sol3 = solve_ivp(
                f, t_span, y0,
                method='Radau',
                rtol=1e-13, atol=1e-15
            )
            if sol3.success:
                y_current = sol3.y[:, -1]
        except Exception:
            pass
    
    # If Radau failed, try BDF as fallback
    if np.allclose(y_current, np.array(y0, dtype=float), rtol=1e-15, atol=1e-15):
        try:
            sol_bdf = solve_ivp(
                f, t_span, y0,
                method='BDF',
                rtol=1e-12, atol=1e-14
            )
            if sol_bdf.success:
                return sol_bdf.y[:, -1]
        except Exception:
            pass
    
    # Final fallback: RK45
    def _rk45_fallback(f, t0, t1, y0, rtol=1e-10, atol=1e-12):
        y = np.array(y0, dtype=float)
        t = t0
        h = (t1 - t0) / 100.0
        while t < t1 and time.time() < deadline:
            h = min(h, t1 - t)
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2*k1)
            k3 = f(t + h/2, y + h/2*k2)
            k4 = f(t + h, y + h*k3)
            y_new = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
            y = y_new
            t += h
            h *= 1.05
        return y
    
    if np.allclose(y_current, np.array(y0, dtype=float), rtol=1e-15, atol=1e-15):
        return _rk45_fallback(f, t0, t1, y0)
    
    return y_current
# EVOLVE-BLOCK-END
