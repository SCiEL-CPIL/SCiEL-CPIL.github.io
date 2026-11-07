"""ODE integrator for the Van der Pol oscillator (mu = 5) using high-order RK."""
import numpy as np


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator from t_span[0] to t_span[1] with initial condition y0.
    Uses 4th-order Runge-Kutta with a large number of steps for high accuracy.
    Returns the state at t_span[1].
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Use RK4 with a large number of steps for high accuracy
    # For Van der Pol with mu=5, the dynamics are smooth enough for explicit RK4
    num_steps = 20000
    h = (t1 - t0) / num_steps
    
    t = t0
    for _ in range(num_steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
    
    return y
