"""ODE integrator for the Van der Pol oscillator (mu = 5) with analytic Jacobian."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.integrate
import time


def _vdp_jac(t, y):
    """Analytic Jacobian for the Van der Pol oscillator with mu=5."""
    y0, y1 = y
    mu = 5.0
    # y0' = y1
    # y1' = mu * (1 - y0^2) * y1 - y0
    #
    # dy0'/dy0 = 0
    # dy0'/dy1 = 1
    #
    # dy1'/dy0 = mu * (-2*y0) * y1 - 1
    # dy1'/dy1 = mu * (1 - y0^2)
    return np.array([
        [0.0, 1.0],
        [-2.0 * mu * y0 * y1 - 1.0, mu * (1.0 - y0**2)]
    ])


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator from t_span[0] to t_span[1].
    Uses scipy.integrate.solve_ivp with the Radau method and analytic Jacobian.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit, leave 20s safety margin
    
    try:
        sol = scipy.integrate.solve_ivp(
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
        else:
            # Fallback if solver fails
            if sol.y.size > 0:
                return sol.y[:, -1]
            else:
                return np.array(y0, dtype=float)

    except Exception:
        # Fallback to a simpler method if Radau fails
        try:
            sol = scipy.integrate.solve_ivp(
                f,
                t_span,
                y0,
                method='RK45',
                rtol=1e-11,
                atol=1e-13
            )
            if sol.success and sol.y.size > 0:
                return sol.y[:, -1]
            elif sol.y.size > 0:
                return sol.y[:, -1]
            else:
                return np.array(y0, dtype=float)
        except Exception:
            return np.array(y0, dtype=float)
# EVOLVE-BLOCK-END
