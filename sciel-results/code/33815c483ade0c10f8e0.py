import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Use BDF (Backward Differentiation Formula) - a multistep method, distinct from Radau (Runge-Kutta)
    # BDF is often efficient for stiff problems and may have different cost/error characteristics.
    sol = solve_ivp(
        f, 
        (t0, t1), 
        y0, 
        method='BDF', 
        rtol=1e-10, 
        atol=1e-13,
        max_step=(t1-t0)/50.0
    )
    
    if sol.success:
        return sol.y[:, -1]
        
    # Fallback to Radau if BDF fails
    sol2 = solve_ivp(
        f, 
        (t0, t1), 
        y0, 
        method='Radau', 
        rtol=1e-10, 
        atol=1e-13,
        max_step=(t1-t0)/50.0
    )
    if sol2.success:
        return sol2.y[:, -1]

    # Last resort: explicit RK4 with many steps
    h = (t1 - t0) / 10000
    t, y = t0, y0.copy()
    for _ in range(10000):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y
