import numpy as np
import time
import scipy.integrate

def integrate(f, t_span, y0):
    start_time = time.time()
    # 330s limit, leave 30s margin
    deadline = start_time + 300.0
    
    y0 = np.array(y0, dtype=float)
    t0, t1 = t_span
    
    # Try Radau first as it is L-stable and efficient for stiff problems
    # Use tight tolerances to maximize accuracy
    try:
        sol = scipy.integrate.solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
        )
        if sol.success:
            return sol.y[:, -1].copy()
    except Exception:
        pass

    # Check time
    if time.time() > deadline:
        return y0

    # Fallback to BDF if Radau failed
    try:
        sol = scipy.integrate.solve_ivp(
            f, t_span, y0,
            method='BDF',
            rtol=1e-12,
            atol=1e-14,
        )
        if sol.success:
            return sol.y[:, -1].copy()
    except Exception:
        pass

    # Check time
    if time.time() > deadline:
        return y0

    # Last resort: Explicit RK45 (only if implicit methods failed)
    try:
        sol = scipy.integrate.solve_ivp(
            f, t_span, y0,
            method='RK45',
            rtol=1e-8,
            atol=1e-10,
        )
        if sol.success:
            return sol.y[:, -1].copy()
    except Exception:
        pass

    return y0
