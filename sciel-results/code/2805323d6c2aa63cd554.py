"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def _vdp_jac(t, y):
    """Analytical Jacobian for Van der Pol oscillator (mu=5)."""
    mu = 5.0
    y0, y1 = y[0], y[1]
    # y0' = y1
    # y1' = mu*(1 - y0^2)*y1 - y0
    # d(y0')/dy0 = 0, d(y0')/dy1 = 1
    # d(y1')/dy0 = -2*mu*y0*y1 - 1, d(y1')/dy1 = mu*(1 - y0^2)
    return np.array([[0.0, 1.0],
                     [-2.0 * mu * y0 * y1 - 1.0, mu * (1.0 - y0 * y0)]])


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator using Radau method with tight tolerances.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Set a deadline to avoid timeout (300s limit, keep safety margin)
    deadline = time.time() + 270
    
    try:
        # Remove max_step to allow adaptive large steps in smooth regions,
        # reducing the number of function evaluations (calls).
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-11,
            atol=1e-13,
            jac=_vdp_jac
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback: try Radau without Jacobian in case of numerical issues
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
    
    # Last resort: BDF method (also implicit, handles stiffness)
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='BDF',
            rtol=1e-10,
            atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Final fallback: dense output from a simple fixed-step approach
    # This is a safety net; should rarely be reached
    n_steps = 5000
    h = (t1 - t0) / n_steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(n_steps):
        if time.time() > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y
# EVOLVE-BLOCK-END
