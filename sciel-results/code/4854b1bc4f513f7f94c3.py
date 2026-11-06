"""ODE solver: Robertson problem using Radau via scipy.integrate.solve_ivp."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns y(t_end).
    
    Uses Radau (implicit Runge-Kutta) with tight tolerances for the stiff
    Robertson problem. Single-stage approach with adaptive step sizing.
    """
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Deadline guard: 320s to leave margin for the evaluator
    deadline = time.time() + 320.0
    
    # Count function calls to monitor budget (max 2,000,000)
    call_count = [0]
    
    def wrapped_f(t, y):
        call_count[0] += 1
        return f(t, y)
    
    # Primary solve: Radau with tight tolerances
    # Radau is an implicit stiff solver that handles the Robertson problem well
    sol1 = solve_ivp(
        wrapped_f, t_span, y0_arr,
        method='Radau',
        rtol=1e-12,
        atol=1e-14
    )
    
    if not sol1.success:
        # Fallback: try with coarser settings
        sol1 = solve_ivp(
            wrapped_f, t_span, y0_arr,
            method='Radau',
            rtol=1e-10,
            atol=1e-12
        )
        if not sol1.success:
            return sol1.y[:, -1]
    
    y_current = sol1.y[:, -1]
    
    # Check time
    if time.time() > deadline - 60.0:
        return y_current
    
    # Stage 2: Refine with tighter tolerance if time permits
    sol2 = solve_ivp(
        wrapped_f, t_span, y0_arr,
        method='Radau',
        rtol=1e-14,
        atol=1e-16
    )
    
    if sol2.success:
        y_current = sol2.y[:, -1]
    
    return y_current
# EVOLVE-BLOCK-END
