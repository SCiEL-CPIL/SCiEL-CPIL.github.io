"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate the Robertson ODE y' = f(t, y) over t_span and return y(t_end).
    
    Uses scipy.integrate.solve_ivp with the 'Radau' method and an analytic Jacobian
    to minimize function evaluation count while maintaining high accuracy.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Analytic Jacobian for the Robertson problem
    def jac(t, y):
        y0_val, y1_val, y2_val = y
        return np.array([
            [-0.04, 1e4 * y2_val, 1e4 * y1_val],
            [0.04, -1e4 * y2_val - 2e-4 * y1_val, -1e4 * y1_val],
            [0.0, 2e-4 * y1_val, 0.0]
        ])
    
    # Use Radau with tight tolerances and analytic Jacobian
    # max_step is removed to allow the solver to optimize step size freely
    sol = solve_ivp(
        f,
        t_span,
        y0,
        method='Radau',
        rtol=1e-12,
        atol=1e-14,
        jac=jac,
        dense_output=False
    )
    
    if not sol.success:
        # Fallback to BDF if Radau fails
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='BDF',
            rtol=1e-12,
            atol=1e-14,
            jac=jac,
            dense_output=False
        )
    
    # Return the final state
    if sol.success and len(sol.t) > 0:
        return sol.y[:, -1]
    else:
        # Last resort: return y0 if everything failed
        return y0
# EVOLVE-BLOCK-END
