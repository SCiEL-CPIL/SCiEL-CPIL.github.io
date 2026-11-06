"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate the Robertson ODE y' = f(t, y) over t_span and return y(t_end).
    
    Uses scipy.integrate.solve_ivp with the 'Radau' method (implicit Runge-Kutta,
    L-stable, suitable for stiff problems) with tight tolerances to minimize
    terminal error while keeping function evaluation count reasonable.
    Provides an analytic Jacobian to reduce function evaluations.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Count function calls by wrapping f
    call_count = [0]
    
    def f_counted(t, y):
        call_count[0] += 1
        return f(t, y)
        
    # Analytic Jacobian for the standard Robertson problem
    # y1' = -0.04*y1 + 1e4*y2*y3
    # y2' = 0.04*y1 - 1e4*y2*y3 - 3e7*y2^2
    # y3' = 3e7*y2^2
    def jac(t, y):
        y1, y2, y3 = y
        return np.array([
            [-0.04, 1e4 * y3, 1e4 * y2],
            [0.04, -1e4 * y3 - 6e7 * y2, -1e4 * y2],
            [0.0, 6e7 * y2, 0.0]
        ])

    # Use Radau with tight tolerances and analytic Jacobian
    # Remove max_step to allow adaptive large steps
    sol = solve_ivp(
        f_counted,
        t_span,
        y0,
        method='Radau',
        rtol=1e-14,
        atol=1e-16,
        jac=jac,
        dense_output=False
    )
    
    if not sol.success:
        # Fallback to BDF if Radau fails
        sol = solve_ivp(
            f_counted,
            t_span,
            y0,
            method='BDF',
            rtol=1e-14,
            atol=1e-16,
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
