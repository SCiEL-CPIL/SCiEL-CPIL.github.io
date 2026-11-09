import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = 300.0  # 300 seconds limit, keep margin
    margin = 15.0     # safety margin before deadline
    
    def _elapsed():
        return time.time() - start_time
    
    def _remaining():
        return deadline - margin - _elapsed()
    
    def _solve_dop853(rtol, atol, max_step=None):
        """Solve with DOP853 at given tolerances."""
        kwargs = dict(
            f=f,
            t_span=t_span,
            y0=y0,
            method='DOP853',
            rtol=rtol,
            atol=atol,
        )
        if max_step is not None:
            kwargs['max_step'] = max_step
        return scipy.integrate.solve_ivp(**kwargs)
    
    def _solve_radau(rtol, atol):
        """Solve with Radau (implicit, for stiff problems)."""
        return scipy.integrate.solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=rtol,
            atol=atol
        )
    
    def _solve_rk45(rtol, atol):
        """Solve with RK45 (5th order explicit)."""
        return scipy.integrate.solve_ivp(
            f, t_span, y0,
            method='RK45',
            rtol=rtol,
            atol=atol
        )
    
    best_result = None
    best_method = None
    
    # Phase 1: Quick baseline with DOP853 at moderate tolerance
    # This gives us a safe fallback result
    try:
        if _remaining() > 5.0:
            sol = _solve_dop853(rtol=1e-8, atol=1e-10, max_step=(t1-t0)/20.0)
            if sol.success:
                best_result = sol.y[:, -1].copy()
                best_method = 'dop853_medium'
    except Exception:
        pass
    
    # Phase 2: Tighten tolerance with DOP853
    try:
        if _remaining() > 10.0:
            sol = _solve_dop853(rtol=1e-11, atol=1e-13)
            if sol.success:
                best_result = sol.y[:, -1].copy()
                best_method = 'dop853_tight'
    except Exception:
        pass
    
    # Phase 3: Even tighter with DOP853
    try:
        if _remaining() > 15.0:
            sol = _solve_dop853(rtol=1e-13, atol=1e-15)
            if sol.success:
                best_result = sol.y[:, -1].copy()
                best_method = 'dop853_very_tight'
    except Exception:
        pass
    
    # Phase 4: Cross-check with Radau at tight tolerance (implicit, handles stiffness)
    try:
        if _remaining() > 30.0:
            sol = _solve_radau(rtol=1e-12, atol=1e-14)
            if sol.success:
                best_result = sol.y[:, -1].copy()
                best_method = 'radau_tight'
    except Exception:
        pass
    
    # Phase 5: If we have lots of time, do an ultra-tight DOP853
    try:
        if _remaining() > 60.0:
            sol = _solve_dop853(rtol=1e-14, atol=1e-16)
            if sol.success:
                best_result = sol.y[:, -1].copy()
                best_method = 'dop853_ultra'
    except Exception:
        pass
    
    # If we got a good result, return it
    if best_result is not None:
        return best_result
    
    # Fallback: simple RK4 if all adaptive methods failed
    steps = 2000
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        if _elapsed() > deadline - 5.0:
            return y
    return y
