"""ODE solver: Robertson problem using DOP853 high-order explicit RK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns y(t_end).
    
    Uses DOP853 (high-order explicit Runge-Kutta) which is efficient for the
    Robertson problem's dynamics, minimizing function evaluations while
    achieving high accuracy through its 8th-order method.
    """
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Deadline guard: 300s to leave margin for the evaluator
    deadline = time.time() + 300.0
    
    # Count function calls to monitor budget (max 2,000,000)
    call_count = [0]
    
    def wrapped_f(t, y):
        call_count[0] += 1
        return f(t, y)
    
    # Primary solve: DOP853 with tight tolerances
    # DOP853 is an 8th-order explicit method with large stability region
    sol = solve_ivp(
        wrapped_f, t_span, y0_arr,
        method='DOP853',
        rtol=1e-13,
        atol=1e-15,
        max_step=np.inf
    )
    
    if not sol.success:
        # Fallback: try with slightly coarser tolerances
        sol = solve_ivp(
            wrapped_f, t_span, y0_arr,
            method='DOP853',
            rtol=1e-11,
            atol=1e-13,
            max_step=np.inf
        )
        if not sol.success:
            # Final fallback: use Radau with Jacobian
            def jac(t, y):
                J = np.array([
                    [-0.04, 1e4*y[2], 1e4*y[1]],
                    [0.04, -1e4*y[2] - 6e7*y[1], -1e4*y[1]],
                    [0.0, 6e7*y[1], 0.0]
                ])
                return J
            
            sol = solve_ivp(
                wrapped_f, t_span, y0_arr,
                method='Radau',
                jac=jac,
                rtol=1e-12,
                atol=1e-14
            )
            return sol.y[:, -1]
    
    y_current = sol.y[:, -1]
    
    # Check if we have time for a refinement pass with tighter tolerance
    if time.time() < deadline - 30.0 and call_count[0] < 1500000:
        sol2 = solve_ivp(
            wrapped_f, t_span, y0_arr,
            method='DOP853',
            rtol=1e-14,
            atol=1e-16,
            max_step=np.inf
        )
        if sol2.success:
            y_current = sol2.y[:, -1]
    
    return y_current
# EVOLVE-BLOCK-END
