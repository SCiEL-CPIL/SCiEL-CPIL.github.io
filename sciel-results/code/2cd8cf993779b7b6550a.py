"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator (mu=5) from t_span[0] to t_span[1]
    with initial condition y0, returning the state at t_span[1].
    
    Uses scipy's Radau implicit Runge-Kutta method which is highly efficient
    for stiff problems like the Van der Pol oscillator with mu=5.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Set a deadline guard: 300s limit, use 290s as safety margin
    deadline = time.time() + 290.0
    
    # Use Radau with tight tolerances matching the reference
    # Radau is an implicit 3rd-order method, excellent for stiff problems
    # The Van der Pol with mu=5 is moderately stiff, so Radau is ideal
    
    result = solve_ivp(
        f,
        t_span=(t0, t1),
        y0=y0,
        method='Radau',
        rtol=1e-11,
        atol=1e-13,
        max_step=0.1  # Ensure reasonable step size for accuracy
    )
    
    if not result.success:
        # Fallback: try with slightly looser tolerances but still very tight
        result = solve_ivp(
            f,
            t_span=(t0, t1),
            y0=y0,
            method='Radau',
            rtol=1e-10,
            atol=1e-12
        )
    
    # Check if we're still within time budget
    if time.time() > deadline - 5.0:
        # Return best result we have
        if result.success:
            return result.y[:, -1]
        else:
            # Last resort: use whatever we have
            if result.y is not None and result.y.shape[1] > 0:
                return result.y[:, -1]
            return y0
    
    if result.success:
        y_end = result.y[:, -1]
        # Verify finite result
        if np.all(np.isfinite(y_end)):
            return y_end
    
    # Fallback if something went wrong: use a simple RK4 as backup
    # This is unlikely to be needed but provides safety
    h = (t1 - t0) / 10000
    t, y = t0, y0.copy()
    for _ in range(10000):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            break
        if time.time() > deadline - 2.0:
            break
    
    return y
# EVOLVE-BLOCK-END
