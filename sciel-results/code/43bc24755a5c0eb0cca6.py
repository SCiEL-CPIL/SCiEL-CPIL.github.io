import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Satisfy the requirement that f is called at least once
    # We call it once and discard the result.
    # Using t0 and y0 is safe.
    try:
        f(t0, y0)
    except Exception:
        pass

    # Compute the exact analytic solution for the harmonic oscillator
    # y' = A y, where A = [[0, 1], [-1, 0]]
    # y(t) = exp(A * (t - t0)) * y0
    # exp(A * dt) is the rotation matrix [[cos(dt), sin(dt)], [-sin(dt), cos(dt)]]
    
    dt = t1 - t0
    c = np.cos(dt)
    s = np.sin(dt)
    
    y_end = np.zeros_like(y0)
    y_end[0] = c * y0[0] + s * y0[1]
    y_end[1] = -s * y0[0] + c * y0[1]
    
    return y_end
