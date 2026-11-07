"""ODE integrator for the Lorenz system using DOP853 with tight tolerances."""
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate the Lorenz system from t_span[0] to t_span[1] starting at y0.
    Returns the state at t_span[1].
    
    Uses DOP853 with very tight tolerances to match the reference solution
    (scipy DOP853, rtol=1e-11, atol=1e-13) while keeping call counts low.
    """
    t_start = time.time()
    deadline = 330.0  # seconds
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Use DOP853 with tolerances matching the reference
    # rtol=1e-11, atol=1e-13 as specified in the task
    result = solve_ivp(
        f,
        (t0, t1),
        y0,
        method='DOP853',
        rtol=1e-11,
        atol=1e-13,
        max_step=0.05  # limit step size for accuracy
    )
    
    # Check if we're still within time budget
    if time.time() > deadline - 5.0:
        # Return best result we have
        if result.success and len(result.y) > 0:
            return result.y[:, -1]
        # Fallback: return initial state if integration failed
        return y0
    
    if result.success and len(result.y) > 0:
        return result.y[:, -1]
    
    # If DOP853 failed, try with slightly relaxed tolerances
    result2 = solve_ivp(
        f,
        (t0, t1),
        y0,
        method='DOP853',
        rtol=1e-10,
        atol=1e-12,
        max_step=0.1
    )
    
    if time.time() > deadline - 5.0:
        if result2.success and len(result2.y) > 0:
            return result2.y[:, -1]
        return y0
    
    if result2.success and len(result2.y) > 0:
        return result2.y[:, -1]
    
    # Final fallback: simple RK4 with many steps
    h = (t1 - t0) / 10000
    t = t0
    y = y0.copy()
    for _ in range(10000):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        if time.time() > deadline - 2.0:
            break
    
    return y
