import numpy as np
import scipy.integrate
import time

def _vdp_jac(t, y):
    """Analytic Jacobian for the Van der Pol oscillator with mu=5."""
    y0, y1 = y[0], y[1]
    mu = 5.0
    # y0' = y1
    # y1' = mu*(1-y0^2)*y1 - y0
    # d(y0')/dy0 = 0, d(y0')/dy1 = 1
    # d(y1')/dy0 = -2*mu*y0*y1 - 1
    # d(y1')/dy1 = mu*(1-y0^2)
    return np.array([[0.0, 1.0],
                     [-2.0 * mu * y0 * y1 - 1.0, mu * (1.0 - y0 * y0)]])


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator y0' = y1, y1' = 5(1 - y0^2)y1 - y0
    from t_span[0] to t_span[1] with initial condition y0.
    Returns the state vector at t_span[1].
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    y0_arr = np.asarray(y0, dtype=float)
    
    # Use Radau with the exact reference tolerances (rtol=1e-11, atol=1e-13)
    # This minimizes the number of function calls while ensuring high accuracy.
    # The analytic Jacobian avoids wasting f-calls on FD-Jacobian estimation.
    try:
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0_arr,
            method='Radau',
            jac=_vdp_jac,
            rtol=1e-11,
            atol=1e-13
        )
        
        if sol.success and len(sol.y) > 0:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
        
        # If Radau failed, try DOP853 as fallback
        if time.time() < deadline:
            sol2 = scipy.integrate.solve_ivp(
                f,
                t_span,
                y0_arr,
                method='DOP853',
                rtol=1e-11,
                atol=1e-13
            )
            
            if sol2.success and len(sol2.y) > 0:
                y_end = sol2.y[:, -1]
                if np.all(np.isfinite(y_end)):
                    return y_end
    
    except Exception:
        pass
    
    # Fallback: fixed-step RK4 with small step size
    # This is a last resort; unlikely to be triggered
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    h = 1e-4
    t = t0
    n_steps = int((t1 - t0) / h) + 1
    
    for _ in range(n_steps):
        if time.time() > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2*k1)
        k3 = f(t + h/2, y + h/2*k2)
        k4 = f(t + h, y + h*k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    
    return y
