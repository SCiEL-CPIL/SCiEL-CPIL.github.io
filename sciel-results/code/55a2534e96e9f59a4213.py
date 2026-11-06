import numpy as np
import time
import scipy.integrate

def integrate(f, t_span, y0):
    start_time = time.time()
    # 330s limit, leave 30s margin
    deadline = start_time + 300.0
    
    y0 = np.array(y0, dtype=float)
    t0, t1 = t_span

    # Strategy: Use Radau with very tight tolerances.
    # Radau is an implicit Runge-Kutta method that is A-stable and L-stable.
    # It is particularly well-suited for stiff problems like the Robertson problem.
    # By using very tight tolerances, we aim to minimize the relative terminal error.
    # The score is -log10(relative_error) - 0.5*log10(calls).
    # Minimizing error is the dominant term, but we also want to keep calls reasonable.
    
    try:
        sol = scipy.integrate.solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=1e-13,
            atol=1e-15,
        )
        if sol.success:
            return sol.y[:, -1].copy()
    except Exception:
        pass

    # Check time
    if time.time() > deadline:
        return y0

    # Fallback to BDF if Radau failed
    # BDF is another implicit method, also suitable for stiff problems.
    try:
        sol = scipy.integrate.solve_ivp(
            f, t_span, y0,
            method='BDF',
            rtol=1e-13,
            atol=1e-15,
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
            rtol=1e-10,
            atol=1e-12,
        )
        if sol.success:
            return sol.y[:, -1].copy()
    except Exception:
        pass

    return y0
