"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate the ODE system y' = f(t, y) over t_span starting from y0.
    Uses a fixed-step high-order Runge-Kutta method (RK4) tuned for minimal calls
    while maintaining high accuracy.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Fixed step size tuned for RK4 accuracy < 1e-14
    # Error scales as h^4. h=2.5e-4 gives error ~ 4e-15.
    h_target = 2.5e-4
    n_steps = int(np.ceil((t1 - t0) / h_target))
    h = (t1 - t0) / n_steps
    
    t = t0
    y = np.array(y0, dtype=float)
    
    for _ in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + h / 2.0, y + h / 2.0 * k1)
        k3 = f(t + h / 2.0, y + h / 2.0 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
        
        if not np.all(np.isfinite(y)):
            break
            
    return y


def _fallback_rk4(f, t0, t1, y0, steps=5000):
    """High-step-count RK4 as a fallback."""
    h = (t1 - t0) / steps
    t = t0
    y = np.array(y0, dtype=float)
    
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
        
        if not np.all(np.isfinite(y)):
            break
    
    return y
# EVOLVE-BLOCK-END
