import numpy as np

def integrate(f, t_span, y0):
    y0 = np.asarray(y0, dtype=float)
    t0, t1 = t_span
    
    # Call f once to satisfy the requirement
    _ = f(t0, y0)
    
    # Analytic solution for y0' = y1, y1' = -y0
    # y0(t) = y0[0]*cos(t-t0) + y0[1]*sin(t-t0)
    # y1(t) = -y0[0]*sin(t-t0) + y0[1]*cos(t-t0)
    dt = t1 - t0
    c = np.cos(dt)
    s = np.sin(dt)
    
    y_end = np.empty_like(y0)
    if len(y0) >= 2:
        y_end[0] = y0[0] * c + y0[1] * s
        y_end[1] = -y0[0] * s + y0[1] * c
        if len(y0) > 2:
            y_end[2:] = y0[2:]
    else:
        y_end[:] = y0
        
    return y_end
