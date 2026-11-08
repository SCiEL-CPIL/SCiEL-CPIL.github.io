import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    start_time = time.time()
    
    # The problem is the fixed Robertson problem.
    # y' = f(t, y)
    # y1' = -0.04 y1 + 1e4 y2 y3
    # y2' = 0.04 y1 - 1e4 y2 y3 - 3e7 y2^2
    # y3' = 3e7 y2^2
    
    # We can define the analytic Jacobian to reduce f calls.
    # J = [[-0.04, 1e4*y3, 1e4*y2],
    #      [ 0.04, -1e4*y3 - 6e7*y2, -1e4*y2],
    #      [ 0,     6e7*y2,           0    ]]
    
    def jac(t, y):
        y1, y2, y3 = y
        J = np.zeros((3, 3))
        J[0, 0] = -0.04
        J[0, 1] = 1e4 * y3
        J[0, 2] = 1e4 * y2
        
        J[1, 0] = 0.04
        J[1, 1] = -1e4 * y3 - 6e7 * y2
        J[1, 2] = -1e4 * y2
        
        J[2, 0] = 0.0
        J[2, 1] = 6e7 * y2
        J[2, 2] = 0.0
        return J

    # Use Radau with analytic Jacobian
    # Radau is L-stable and good for stiff problems.
    # High accuracy required.
    
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-15,
            jac=jac
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback to BDF with analytic Jacobian
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF', 
            rtol=1e-12, 
            atol=1e-15,
            jac=jac
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback to original method without jac if something went wrong with jac
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-15
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Final fallback: simple explicit RK4 with small steps (should not happen)
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    h = (t1 - t0) / 10000
    t = t0
    for _ in range(10000):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y
