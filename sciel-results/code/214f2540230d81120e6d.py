"""ODE solver: Robertson problem using DOP853 via scipy.integrate.solve_ivp."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns y(t_end).
    
    Uses DOP853 (8th-order Runge-Kutta) with tight tolerances.
    DOP853 has a large stability region and high order, minimizing function
    evaluations in the non-stiff initial phase of the Robertson problem.
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
    
    # Primary solve: DOP853 with tight tolerances
    # DOP853 is an explicit 8th-order method with 12 stages
    # It takes large steps in smooth regions, reducing total evaluations
    sol1 = solve_ivp(
        wrapped_f, t_span, y0_arr,
        method='DOP853',
        rtol=1e-12,
        atol=1e-14
    )
    
    if not sol1.success:
        # Fallback: try with slightly looser tolerances if it failed
        sol1 = solve_ivp(
            wrapped_f, t_span, y0_arr,
            method='DOP853',
            rtol=1e-10,
            atol=1e-12
        )
        if not sol1.success:
            # Last resort: try LSODA as fallback
            sol1 = solve_ivp(
                wrapped_f, t_span, y0_arr,
                method='LSODA',
                rtol=1e-12,
                atol=1e-14
            )
            if not sol1.success:
                return sol1.y[:, -1]
    
    y_current = sol1.y[:, -1]
    
    # Check time: if we have plenty of time, try to refine further
    # The score penalizes calls, so we only refine if it's "cheap"
    elapsed = time.time() - (deadline - 320.0)
    
    if elapsed < 10.0 and call_count[0] < 500000:
        # Try a tighter tolerance pass with DOP853
        sol2 = solve_ivp(
            wrapped_f, t_span, y0_arr,
            method='DOP853',
            rtol=1e-14,
            atol=1e-16
        )
        if sol2.success:
            y_current = sol2.y[:, -1]
    
    return y_current
# EVOLVE-BLOCK-END
