"""ODE integrator for the Lotka-Volterra system using a high-order Runge-Kutta method."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system y0' = 1.5*y0 - y0*y1, y1' = -3*y1 + y0*y1
    from t_span[0] to t_span[1] with initial condition y0.
    
    Returns the state at the end of t_span.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # We know the exact ODE, so we can use a very accurate method.
    # The key insight: we can use scipy's DOP853 with very tight tolerances,
    # but we need to be careful about the number of f calls.
    
    # Actually, let's use a high-order Runge-Kutta method with adaptive stepping.
    # But first, let's try a simple approach: use many small steps with RK4.
    # The Lotka-Volterra system is not stiff, so explicit methods work well.
    
    # Let's use a fixed-step high-order method. DOP853 is 8th order.
    # But implementing DOP853 from scratch is complex. Let's use scipy instead.
    
    from scipy.integrate import solve_ivp
    
    # Use DOP853 with very tight tolerances
    sol = solve_ivp(f, t_span, y0, method='DOP853', rtol=1e-13, atol=1e-15, max_step=0.01)
    
    if sol.success:
        return sol.y[:, -1]
    else:
        # Fallback: use RK4 with many steps
        h = (t1 - t0) / 10000
        t = t0
        for _ in range(10000):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2*k1)
            k3 = f(t + h/2, y + h/2*k2)
            k4 = f(t + h, y + h*k3)
            y = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y
# EVOLVE-BLOCK-END
