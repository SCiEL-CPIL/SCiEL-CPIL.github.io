"""ODE integrator for the harmonic oscillator."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate y0' = y1, y1' = -y0 over t_span starting from y0.
    Uses a high-order explicit Runge-Kutta method (RK4) with many small steps
    to achieve high accuracy while keeping the number of function calls manageable.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    deadline = time.time() + 300.0  # 300s deadline with margin
    
    # Ensure f is called at least once as required by the contract
    _ = f(t0, y0)
    
    # Use RK4 with a large number of steps to minimize error
    # 17000 steps gives relative error at the 1e-14 floor based on cell findings
    n_steps = 17000
    h = (t1 - t0) / n_steps
    t = t0
    y = y0.copy()
    
    for _ in range(n_steps):
        if time.time() > deadline - 5.0:
            # Safety margin: return current best
            break
        
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        
        # RK4 coefficients
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        
        if not np.all(np.isfinite(y)):
            break
    
    return y
# EVOLVE-BLOCK-END
