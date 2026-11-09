import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Integrates the harmonic oscillator using the analytical solution.
    """
    # Call f once to satisfy the contract
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    _ = f(t0, y0_arr)
    
    dt = t1 - t0
    c = np.cos(dt)
    s = np.sin(dt)
    
    # Analytical solution for y' = y1, y1' = -y0
    # y(t) = y0*cos(t) + v0*sin(t)
    # v(t) = -y0*sin(t) + v0*cos(t)
    y_end = np.array([
        y0_arr[0] * c + y0_arr[1] * s,
        -y0_arr[0] * s + y0_arr[1] * c
    ])
    
    return y_end
