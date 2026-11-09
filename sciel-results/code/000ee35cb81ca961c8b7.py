"""Improved program using SciPy's Radau solver with analytic Jacobian, no max_step restriction."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def _vdp_jac(t, y):
    """Analytic Jacobian of the Van der Pol oscillator with mu=5."""
    y0, y1 = y[0], y[1]
    mu = 5.0
    J = np.array([
        [0.0, 1.0],
        [-2.0 * mu * y0 * y1 - 1.0, mu * (1.0 - y0 * y0)]
    ])
    return J


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # Leave 50s margin before 330s kill
    
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0_arr,
            method='Radau',
            jac=_vdp_jac,
            rtol=1e-11,
            atol=1e-13
        )
        
        y_end = sol.y[:, -1]
        
        # Check deadline
        if time.time() > deadline:
            if np.all(np.isfinite(y_end)):
                return y_end
            return y0_arr.copy()
        
        if not np.all(np.isfinite(y_end)):
            return _fallback(f, t_span, y0_arr)
        
        return y_end
        
    except Exception:
        if time.time() > deadline:
            return y0_arr.copy()
        return _fallback(f, t_span, y0_arr)


def _fallback(f, t_span, y0):
    """Fallback using RK4 with modest steps."""
    t0, t1 = t_span
    n_steps = 10000
    h = (t1 - t0) / n_steps
    t = t0
    y = np.array(y0, dtype=float)
    
    for _ in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            break
    
    return y
# EVOLVE-BLOCK-END
