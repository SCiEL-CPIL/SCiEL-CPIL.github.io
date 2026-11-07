"""High-order explicit RK via SciPy solve_ivp (DOP853) for Van der Pol."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.integrate as integrate_scipy

def integrate(f, t_span, y0):
    """
    Integrate the ODE using scipy.integrate.solve_ivp with DOP853 (8th order)
    or fallback to RK45.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Try DOP853 first (8th order, 12 stages)
    try:
        sol = integrate_scipy.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-12, 
            atol=1e-14
        )
        if sol.success and sol.y.shape[1] > 0:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback to RK45 (5th order, 6 stages)
    try:
        sol = integrate_scipy.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='RK45', 
            rtol=1e-12, 
            atol=1e-14
        )
        if sol.success and sol.y.shape[1] > 0:
            return sol.y[:, -1]
    except Exception:
        pass

    # Final fallback: Manual RK4 with fixed steps
    N = 10000
    h = (t1 - t0) / N
    y = y0.copy()
    t = t0
    for _ in range(N):
        k1 = np.asarray(f(t, y), dtype=float)
        k2 = np.asarray(f(t + 0.5*h, y + 0.5*h*k1), dtype=float)
        k3 = np.asarray(f(t + 0.5*h, y + 0.5*h*k2), dtype=float)
        k4 = np.asarray(f(t + h, y + h*k3), dtype=float)
        y = y + (h/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        t += h
    return y

# EVOLVE-BLOCK-END
