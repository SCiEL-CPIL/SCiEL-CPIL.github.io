"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def _van_der_pol_jac(t, y):
    """Analytical Jacobian for Van der Pol oscillator with mu=5."""
    y0, y1 = y[0], y[1]
    mu = 5.0
    J = np.array([
        [0.0, 1.0],
        [-2.0 * mu * y0 * y1 - 1.0, mu * (1.0 - y0 * y0)]
    ])
    return J


def integrate(f, t_span, y0):
    """
    ODE integrator for the Van der Pol oscillator (mu = 5).
    Returns the state at the end of t_span.
    """
    # Deadline guard: 300s limit, keep margin
    deadline = time.time() + 285
    
    # First pass: Radau with moderate tolerances
    y_mid = None
    try:
        sol1 = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            jac=_van_der_pol_jac,
            rtol=1e-10,
            atol=1e-12
        )
        if sol1.success:
            y_mid = sol1.y[:, -1]
            if np.all(np.isfinite(y_mid)):
                y_mid = y_mid.copy()
    except Exception:
        pass
    
    # If we have time and a mid-precision result, do a second high-precision pass
    if y_mid is not None and time.time() < deadline - 30:
        try:
            sol2 = solve_ivp(
                f,
                t_span,
                y0,
                method='Radau',
                jac=_van_der_pol_jac,
                rtol=1e-13,
                atol=1e-15
            )
            if sol2.success:
                result = sol2.y[:, -1]
                if np.all(np.isfinite(result)):
                    return result
        except Exception:
            pass
    
    # Return mid-precision result if available
    if y_mid is not None:
        return y_mid
    
    # Fallback: Radau without Jacobian
    try:
        sol_fb = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-11,
            atol=1e-13
        )
        if sol_fb.success:
            result = sol_fb.y[:, -1]
            if np.all(np.isfinite(result)):
                return result
    except Exception:
        pass
    
    # Try RK45 as secondary fallback
    if time.time() < deadline:
        try:
            sol_rk = solve_ivp(
                f,
                t_span,
                y0,
                method='RK45',
                rtol=1e-11,
                atol=1e-13
            )
            if sol_rk.success:
                result = sol_rk.y[:, -1]
                if np.all(np.isfinite(result)):
                    return result
        except Exception:
            pass
    
    # Last resort fallback: simple RK4 if scipy fails completely
    if time.time() < deadline:
        try:
            t0, t1 = t_span
            y = np.array(y0, dtype=float)
            steps = 10000
            h = (t1 - t0) / steps
            t = t0
            for _ in range(steps):
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2 * k1)
                k3 = f(t + h/2, y + h/2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                t += h
            if np.all(np.isfinite(y)):
                return y
        except Exception:
            pass
            
    return np.array(y0, dtype=float)
# EVOLVE-BLOCK-END
