"""Rewritten ODE integrator for the Lotka-Volterra system using Fixed-Step 4th Order Runge-Kutta (RK4) with a small step size to balance accuracy and function call count, distinct from adaptive methods."""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    """
    Integrate the fixed Lotka-Volterra IVP from t_span[0] to t_span[1] starting at y0.
    Uses a fixed-step 4th-order Runge-Kutta method.
    """
    t0, t1 = t_span
    if t1 <= t0:
        return np.array(y0, dtype=float)

    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Choose a small fixed step size
    # h = 2e-4 gives ~25,000 steps, ~100,000 calls.
    # Error is ~ (2e-4)^4 = 1.6e-15, which is below the 1e-14 floor.
    h = 2e-4
    t = t0
    
    # Calculate number of steps
    # We want to hit t1 exactly if possible, or handle the last step.
    n_steps = int(round((t1 - t0) / h))
    # Adjust h to exactly reach t1
    h = (t1 - t0) / n_steps
    
    for _ in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)
        
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h

    return y
# EVOLVE-BLOCK-END
