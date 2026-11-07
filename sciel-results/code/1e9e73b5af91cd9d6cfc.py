import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    # Fixed-step RK4 with 1000 steps.
    # This approach avoids adaptive step overhead and ensures predictable error/call count.
    # 1000 steps * 4 calls/step = 4000 calls.
    # Error scales as h^4, so 1000 steps gives ~1e-12 relative error for this system.
    steps = 1000
    h = (t1 - t0) / steps
    t = t0
    
    for i in range(steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        
    return y
