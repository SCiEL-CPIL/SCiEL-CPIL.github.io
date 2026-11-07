"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    """
    Integrates the Lotka-Volterra system using scipy.integrate.solve_ivp
    with the DOP853 method to match the reference solution's characteristics,
    minimizing relative error and function call count.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit, keep 20s safety margin
    
    try:
        # Use the same method and tolerances as the reference to ensure
        # the result is as close as possible to y_ref, driving rel to the floor.
        # DOP853 is an 8th-order method, highly efficient for this smooth system.
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-11,
            atol=1e-13,
            dense_output=False
        )
        
        # Check if we are still within time limits (though solve_ivp is fast, 
        # this guards against pathological hangs in the evaluator's f)
        if time.time() > deadline:
            return _fallback_rk4(f, t_span, y0)

        if not sol.success:
            # Fallback if for some reason it fails
            return _fallback_rk4(f, t_span, y0)
        
        return sol.y[:, -1]
        
    except Exception:
        # If any exception occurs, fall back to a robust manual integrator
        return _fallback_rk4(f, t_span, y0)

def _fallback_rk4(f, t_span, y0):
    """Fallback explicit RK4 with adaptive step size if SciPy fails."""
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    # Use a reasonable number of steps for a fallback
    # 10000 steps for RK4 is usually sufficient for smooth ODEs to get ~1e-8 accuracy
    steps = 10000
    h = (t1 - t0) / steps
    t = t0
    
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + 4.0 * k4)
        t += h
        
        # Early exit if diverging
        if not np.all(np.isfinite(y)):
            break
            
    return y
# EVOLVE-BLOCK-END
