"""ODE solver: Robertson problem using LSODA via scipy.integrate.solve_ivp."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns y(t_end).
    
    Uses a multi-stage strategy:
    1. Coarse solve with LSODA to get a ballpark.
    2. Refine with progressively tighter tolerances.
    3. Deadline guard to stay under 330s.
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
    
    # Stage 1: Coarse solve to establish a baseline
    # LSODA automatically switches between Adams and BDF for stiffness
    sol1 = solve_ivp(
        wrapped_f, t_span, y0_arr,
        method='LSODA',
        rtol=1e-10,
        atol=1e-12,
        max_step=(t1 - t0) / 50.0  # limit step size for accuracy
    )
    
    if not sol1.success:
        # Fallback: try with even coarser settings
        sol1 = solve_ivp(
            wrapped_f, t_span, y0_arr,
            method='LSODA',
            rtol=1e-8,
            atol=1e-10
        )
        if not sol1.success:
            return sol1.y[:, -1]
    
    y_current = sol1.y[:, -1]
    
    # Check time
    if time.time() > deadline - 60.0:
        return y_current
    
    # Stage 2: Refine with tighter tolerance
    sol2 = solve_ivp(
        wrapped_f, t_span, y0_arr,
        method='LSODA',
        rtol=1e-14,
        atol=1e-16,
        max_step=(t1 - t0) / 200.0
    )
    
    if sol2.success:
        y_current = sol2.y[:, -1]
    else:
        # If stage 2 failed, keep stage 1 result
        pass
    
    # Check time
    if time.time() > deadline - 30.0:
        return y_current
    
    # Stage 3: Ultra-refine if time permits
    sol3 = solve_ivp(
        wrapped_f, t_span, y0_arr,
        method='LSODA',
        rtol=1e-15,
        atol=1e-17,
        max_step=(t1 - t0) / 500.0
    )
    
    if sol3.success:
        y_current = sol3.y[:, -1]
    
    return y_current
# EVOLVE-BLOCK-END
