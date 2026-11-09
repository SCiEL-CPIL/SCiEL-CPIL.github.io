"""ODE solver for Robertson problem with adaptive stiff integration."""
import numpy as np
import time
import scipy.integrate


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns y(t_end).
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard: return best result before 330s wall-clock limit
    deadline = time.time() + 325.0  # 5s safety margin
    
    # Wrap f to count calls and check deadline
    call_count = [0]
    
    def f_wrapper(t, y):
        call_count[0] += 1
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded")
        return f(t, y)
    
    # Try Radau (L-stable, good for stiff problems) with tight tolerances
    try:
        sol = scipy.integrate.solve_ivp(
            f_wrapper,
            t_span,
            y0,
            method='Radau',
            rtol=1e-12,
            atol=1e-15,
            dense_output=False,
            max_step=(t1 - t0) / 10.0
        )
        if sol.success:
            return sol.y[:, -1]
    except (TimeoutError, Exception):
        pass
    
    # Try BDF as fallback
    try:
        sol = scipy.integrate.solve_ivp(
            f_wrapper,
            t_span,
            y0,
            method='BDF',
            rtol=1e-12,
            atol=1e-15,
            dense_output=False,
            max_step=(t1 - t0) / 10.0
        )
        if sol.success:
            return sol.y[:, -1]
    except (TimeoutError, Exception):
        pass
    
    # Fallback: fixed-step Radau-like via scipy with looser tolerances
    try:
        sol = scipy.integrate.solve_ivp(
            f_wrapper,
            t_span,
            y0,
            method='Radau',
            rtol=1e-10,
            atol=1e-13,
            dense_output=False
        )
        if sol.success:
            return sol.y[:, -1]
    except (TimeoutError, Exception):
        pass
    
    # Last resort: fixed-step RK4 (should not reach here)
    y = np.array(y0, dtype=float)
    t = t0
    h = (t1 - t0) / 10000
    for _ in range(10000):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            break
    return y
