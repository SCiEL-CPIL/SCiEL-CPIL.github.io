"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def integrate(f, t_span, y0):
    """
    Integrates the ODE y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses a fixed-step Runge-Kutta 4th order method.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    T = t1 - t0
    
    # Choose number of steps to balance accuracy and call count.
    # RK4 global error is O(h^4).
    # For T=10, N=100000 gives h=1e-4, h^4=1e-16.
    # This should push relative error to machine precision floor.
    N = 100000
    h = T / N
    
    for _ in range(N):
        k1 = f(t0, y)
        k2 = f(t0 + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t0 + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t0 + h, y + h * k3)
        
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t0 += h
        
    return y

# EVOLVE-BLOCK-END
