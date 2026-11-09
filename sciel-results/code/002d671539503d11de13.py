"""Rewritten program using SciPy's Radau method with analytic Jacobian for the Robertson problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Solves the Robertson ODE problem using scipy.integrate.solve_ivp with the Radau method.
    Provides an analytic Jacobian to minimize function evaluations of f.
    """
    start_time = time.time()
    # Keep a safety margin before the 300s limit
    deadline = start_time + 280.0
    
    y0 = np.asarray(y0, dtype=float)
    t0, t1 = t_span

    # Analytic Jacobian for the Robertson problem
    # y' = f(t, y)
    # f0 = -0.04 y0 + 1e4 y1 y2
    # f1 =  0.04 y0 - 1e4 y1 y2 - 3e3 y1^2
    # f2 =  3e3 y1^2
    def jac(t, y):
        y0_val, y1_val, y2_val = y
        J = np.array([
            [-0.04, 1e4 * y2_val, 1e4 * y1_val],
            [0.04, -1e4 * y2_val - 6e3 * y1_val, -1e4 * y1_val],
            [0.0, 6e3 * y1_val, 0.0]
        ])
        return J

    try:
        # Use Radau (implicit RK, stiff solver) with tight tolerances
        # The analytic Jacobian avoids finite-difference calls to f, reducing the 'evals' penalty
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            jac=jac,
            rtol=1e-14,
            atol=1e-16,
            # max_step can be left to the solver's adaptive logic
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback if Radau fails
            return _fallback_integrate(f, t_span, y0, deadline)

    except Exception:
        return _fallback_integrate(f, t_span, y0, deadline)


def _fallback_integrate(f, t_span, y0, deadline):
    """
    Fallback using BDF without analytic Jacobian if the primary method fails.
    """
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
    
    # Last resort: return initial state (will score 0, but prevents crash)
    return y0
# EVOLVE-BLOCK-END
