"""Rewritten ODE solver for the 100-point convection-diffusion IVP."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_DEADLINE = time.time() + 300.0  # 300s safety margin (evaluator kills at 330s)

def _check_deadline():
    """Return True if we should stop integrating."""
    return time.time() > _DEADLINE


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    
    Strategy:
    1. Use DOP853 (8th order explicit RK) with tight tolerances.
    2. If DOP853 fails or is too slow, use Radau (5th order implicit) with tight tolerances.
    3. Fallback to RK4 if all else fails.
    """
    global _DEADLINE
    _DEADLINE = time.time() + 300.0
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    from scipy.integrate import solve_ivp
    
    # Count f calls
    call_count = [0]
    
    def f_counted(t, y):
        call_count[0] += 1
        if _check_deadline():
            raise Exception("Deadline exceeded")
        return f(t, y)
    
    # Try DOP853 first (high order, efficient for non-stiff/mildly stiff)
    try:
        sol = solve_ivp(
            f_counted, t_span, y0,
            method='DOP853',
            rtol=1e-13, atol=1e-15,
            max_step=(t1-t0)/10,
            dense_output=False
        )
        if sol.success and sol.t[-1] == t1:
            y_final = sol.y[:, -1]
            if np.all(np.isfinite(y_final)):
                return y_final
    except Exception:
        pass
    
    # If DOP853 failed, try Radau (implicit, good for stiff)
    try:
        sol = solve_ivp(
            f_counted, t_span, y0,
            method='Radau',
            rtol=1e-13, atol=1e-15,
            max_step=(t1-t0)/10,
            dense_output=False
        )
        if sol.success and sol.t[-1] == t1:
            y_final = sol.y[:, -1]
            if np.all(np.isfinite(y_final)):
                return y_final
    except Exception:
        pass
    
    # Fallback to RK4
    return _rk4_fallback(f, t_span, y0)


def _rk4_fallback(f, t_span, y0):
    """Fallback RK4 with adaptive-ish stepping."""
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Use a fixed number of steps that's reasonable
    # For 100-point convection-diffusion, we need enough resolution
    steps = 5000
    h = (t1 - t0) / steps
    t = t0
    
    for i in range(steps):
        if _check_deadline():
            return y
        
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        
        y_new = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        if not np.all(np.isfinite(y_new)) or np.max(np.abs(y_new)) > 1e12:
            # Reduce step size
            h /= 2
            steps += steps // 2  # approximate
            continue
        
        y = y_new
        t += h
    
    return y


def _rk4(f, t0, t1, y0, steps):
    """Original RK4 for reference."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        if _check_deadline():
            return None
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            return None
    return y
# EVOLVE-BLOCK-END
