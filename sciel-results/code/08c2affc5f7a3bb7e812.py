"""Optimized DOP853 integrator for Lotka-Volterra system using SciPy."""
import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrates the Lotka-Volterra system from t_span[0] to t_span[1].
    Returns the state at the end of t_span.
    """
    y0_arr = np.asarray(y0, dtype=float)
    t0, t1 = t_span
    
    if abs(t1 - t0) < 1e-15:
        return y0_arr.copy()
    
    # Use DOP853 with tolerances matching the reference to minimize calls
    # while keeping relative error bounded by the reference's error.
    # Reference: rtol=1e-11, atol=1e-13.
    # Using slightly tighter rtol=1e-12 might not reduce rel (bounded by ref error)
    # but increases calls. So match reference tolerances.
    
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0_arr,
            method='DOP853',
            rtol=1e-11,
            atol=1e-13
        )
        
        if not sol.success:
            # Fallback to tighter tolerance if it fails for some reason (unlikely)
            sol = solve_ivp(
                f,
                t_span,
                y0_arr,
                method='DOP853',
                rtol=1e-12,
                atol=1e-14
            )
            if not sol.success:
                raise RuntimeError("Integration failed")
                
        y_end = sol.y[:, -1]
        
        if not np.all(np.isfinite(y_end)):
            raise RuntimeError("Non-finite result")
            
        return y_end
        
    except Exception:
        # Last resort fallback: simple RK4 with many steps
        # This is very expensive in calls, so only if SciPy fails
        t, y = t0, y0_arr.copy()
        n_steps = 10000
        h = (t1 - t0) / n_steps
        for i in range(n_steps):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y
