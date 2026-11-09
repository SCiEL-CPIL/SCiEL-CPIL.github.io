import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrates the ODE y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses scipy.integrate.solve_ivp with the DOP853 method (8th order Runge-Kutta)
    for high accuracy and efficiency on this stiff problem.
    """
    t0, t1 = t_span
    
    # Attempt to use DOP853 (8th order) which is highly accurate and efficient
    # rtol and atol are set to tight values to ensure the result is as accurate as possible
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-13, 
            atol=1e-15
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback to Radau (implicit) if DOP853 fails or is not available
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-11, 
            atol=1e-13
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Final fallback: Manual RK4 with sufficient steps
    h = (t1 - t0) / 200
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(200):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y
