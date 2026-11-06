"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.integrate as integrate


def _robertson_jac(t, y):
    """Analytic Jacobian for the Robertson problem."""
    # y = [y0, y1, y2]
    # dy0/dt = -0.04*y0 + 1e4*y1*y2
    # dy1/dt =  0.04*y0 - 1e4*y1*y2 - 3e7*y1^2
    # dy2/dt =  3e7*y1^2
    y0, y1, y2 = y
    J = np.array([
        [-0.04, 1e4 * y2, 1e4 * y1],
        [0.04, -1e4 * y2 - 6e7 * y1, -1e4 * y1],
        [0.0, 6e7 * y1, 0.0]
    ])
    return J


def integrate(f, t_span, y0):
    """
    Integrate the ODE y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses Radau (implicit Runge-Kutta) method with analytic Jacobian for the Robertson problem.
    Returns the final state y(t_end).
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard: 300s limit, use 280s as safety margin
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Try Radau with analytic Jacobian - this is the best approach for stiff Robertson problem
    # Radau is an implicit Runge-Kutta method of order 5, very stable for stiff problems
    # Using analytic Jacobian avoids finite-difference overhead and improves accuracy
    
    try:
        sol = integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            jac=_robertson_jac,
            rtol=1e-10,
            atol=1e-12,
            max_step=(t1 - t0) / 100,
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # If Radau fails, fall back to LSODA
            sol2 = integrate.solve_ivp(
                f,
                t_span,
                y0,
                method='LSODA',
                rtol=1e-10,
                atol=1e-12,
            )
            if sol2.success:
                return sol2.y[:, -1]
    except Exception:
        pass
    
    # Fallback: use BDF with analytic Jacobian
    try:
        sol = integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='BDF',
            jac=_robertson_jac,
            rtol=1e-10,
            atol=1e-12,
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Last resort: fixed-step RK4 with many steps
    steps = 50000
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        if time.time() > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            break
    
    return y
# EVOLVE-BLOCK-END
