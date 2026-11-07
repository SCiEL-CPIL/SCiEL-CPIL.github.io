"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Integrate the Lorenz system from t_span[0] to t_span[1] starting at y0.
    Returns the state at t_span[1].
    
    Strategy: Use scipy's DOP853 with the evaluator's f and very tight tolerances
    to achieve near-machine-precision accuracy. The number of f calls is moderate
    (~1000-5000), but the accuracy gain far outweighs the call penalty.
    """
    from scipy.integrate import solve_ivp
    
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s limit
    
    y0 = np.array(y0, dtype=float)
    
    # Use DOP853 with very tight tolerances for high accuracy
    sol = solve_ivp(
        f,
        t_span,
        y0,
        method='DOP853',
        rtol=1e-13,
        atol=1e-15,
    )
    
    if sol.success:
        y_end = sol.y[:, -1]
    else:
        # Fallback: use a simple RK4 with many steps
        y = y0.copy()
        t0, t1 = t_span
        n_steps = 1000
        dt = (t1 - t0) / n_steps
        for i in range(n_steps):
            if time.time() > deadline:
                break
            t = t0 + i * dt
            k1 = f(t, y)
            k2 = f(t + dt/2.0, y + dt/2.0 * k1)
            k3 = f(t + dt/2.0, y + dt/2.0 * k2)
            k4 = f(t + dt, y + dt * k3)
            y = y + dt/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)
        y_end = y
    
    return np.array(y_end, dtype=float)
# EVOLVE-BLOCK-END
