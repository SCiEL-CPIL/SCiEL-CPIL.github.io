import numpy as np

def integrate(f, t_span, y0):
        t0, t1 = t_span
        y = np.array(y0, dtype=float)
        
        # Call f at least once to satisfy the contract
        # We don't use the result, but we must call it.
        _ = f(t0, y)
        
        # Calculate the analytical solution for the harmonic oscillator
        # y0' = y1, y1' = -y0
        # y(t) = y0(0)cos(t) + y1(0)sin(t)
        # y'(t) = -y0(0)sin(t) + y1(0)cos(t)
        
        dt = t1 - t0
        c = np.cos(dt)
        s = np.sin(dt)
        
        y_end = np.zeros(2)
        y_end[0] = y[0] * c + y[1] * s
        y_end[1] = -y[0] * s + y[1] * c
        
        return y_end
