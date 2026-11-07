import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # RK4 with fixed step
    # dt chosen to balance error and call count
    # Reference error ~ 1e-11. RK4 error ~ dt^4.
    # dt=1e-3 -> error ~ 1e-12 (good)
    # dt=2e-3 -> error ~ 1.6e-11 (borderline)
    # We use 1e-3 for safety.
    dt = 1e-3
    n_steps = int(np.ceil((t1 - t0) / dt))
    dt = (t1 - t0) / n_steps
    
    t = t0
    for _ in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = f(t + dt, y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt
        
    return y
