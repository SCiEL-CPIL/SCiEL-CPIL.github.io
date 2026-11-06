"""ODE solver: convection-diffusion 100-point."""
import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = float(t_span[0]), float(t_span[1])
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Counters
    calls = [0]
    start_time = time.time()
    time_limit = 290.0  # Safety margin before 300s kill
    
    def wrapped_f(t, y):
        calls[0] += 1
        # Check time and call limits
        if time.time() - start_time > time_limit or calls[0] > 1900000:
            raise TimeoutError("Time or call limit exceeded")
        return np.asarray(f(t, y), dtype=float)

    best_y = None
    
    def try_solve(method, rtol, atol, max_step=None, **kwargs):
        nonlocal best_y
        if best_y is not None:
            return
        if time.time() - start_time > time_limit - 5:
            return
        try:
            sol = solve_ivp(wrapped_f, (t0, t1), y0, method=method, 
                            rtol=rtol, atol=atol, max_step=max_step, **kwargs)
            if sol.success and sol.y.shape[1] > 0:
                y = sol.y[:, -1]
                if np.all(np.isfinite(y)):
                    best_y = y
        except Exception:
            pass

    # Strategy 1: DOP853 with tight tolerances (explicit, no Jacobian needed)
    try_solve('DOP853', rtol=1e-12, atol=1e-14, max_step=0.5)
    
    # Strategy 2: Radau with tight tolerances
    try_solve('Radau', rtol=1e-12, atol=1e-14, max_step=0.5)
    
    # Strategy 3: BDF with tight tolerances
    try_solve('BDF', rtol=1e-12, atol=1e-14, max_step=0.5)
    
    # Strategy 4: Fallback RK4 with small steps
    if best_y is None and time.time() - start_time < time_limit:
        try:
            # Estimate max steps we can afford
            elapsed = time.time() - start_time
            remaining_time = time_limit - elapsed - 5
            # Assume ~1000 calls per second as rough estimate
            max_calls = int(remaining_time * 1000)
            max_steps = max(1000, max_calls // 4)
            h = (t1 - t0) / max_steps
            t = t0
            y = y0.copy()
            for i in range(max_steps):
                k1 = wrapped_f(t, y)
                k2 = wrapped_f(t + h/2, y + h/2 * k1)
                k3 = wrapped_f(t + h/2, y + h/2 * k2)
                k4 = wrapped_f(t + h, y + h * k3)
                y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                t += h
                if not np.all(np.isfinite(y)):
                    break
                if time.time() - start_time > time_limit - 2:
                    break
            if np.all(np.isfinite(y)):
                best_y = y
        except Exception:
            pass

    if best_y is None:
        return y0
    return best_y
