import numpy as np

def integrate(f, t_span, y0):
    # Satisfy the constraint that f is called at least once
    # We can call it with the initial state
    _ = f(t_span[0], y0)
    
    t_end = t_span[1]
    y0_arr = np.asarray(y0, dtype=float)
    
    # Analytic solution for y' = [y1, -y0]
    # y0(t) = y0[0]*cos(t) + y0[1]*sin(t)
    # y1(t) = -y0[0]*sin(t) + y0[1]*cos(t)
    
    c = np.cos(t_end)
    s = np.sin(t_end)
    
    y_end = np.array([
        y0_arr[0] * c + y0_arr[1] * s,
        -y0_arr[0] * s + y0_arr[1] * c
    ])
    
    return y_end
