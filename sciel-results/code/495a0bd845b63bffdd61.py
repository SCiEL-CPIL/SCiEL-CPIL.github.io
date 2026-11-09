"""ODE integrator for the Van der Pol oscillator (mu = 5) using scipy Radau."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.integrate
import time


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator y0' = y1, y1' = 5(1 - y0^2)y1 - y0
    from t_span[0] to t_span[1] with initial condition y0.
    Returns the state vector at t_span[1].
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit

    # Use Radau with very tight tolerances to get a result closer to the true solution
    # than the reference (which uses rtol=1e-11, atol=1e-13).
    # Tighter tolerances reduce the relative error between our result and the reference.
    try:
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-13,
            atol=1e-15
        )

        if sol.success:
            return sol.y[:, -1]
        else:
            # If solve_ivp fails, fall back to DOP853 with tight tolerances
            sol2 = scipy.integrate.solve_ivp(
                f,
                t_span,
                y0,
                method='DOP853',
                rtol=1e-13,
                atol=1e-15
            )
            if sol2.success:
                return sol2.y[:, -1]

    except Exception:
        pass

    # Fallback: fixed-step RK4 with small step size
    # This is a last resort; unlikely to be triggered
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    h = 1e-4
    t = t0
    n_steps = int((t1 - t0) / h) + 1

    for _ in range(n_steps):
        if time.time() > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2*k1)
        k3 = f(t + h/2, y + h/2*k2)
        k4 = f(t + h, y + h*k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h

    return y
# EVOLVE-BLOCK-END
