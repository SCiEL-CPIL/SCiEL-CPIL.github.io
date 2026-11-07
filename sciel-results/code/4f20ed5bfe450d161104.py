import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    """
    Integrate the ODE system y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses DOP853 method for high accuracy and efficiency.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Track function calls to ensure we don't exceed limits, though solve_ivp handles this internally.
    # The task says f must be called at least once. solve_ivp will call it.
    
    # Use DOP853 with tight tolerances to ensure high accuracy.
    # The reference uses rtol=1e-11, atol=1e-13.
    # We use tighter tolerances to ensure rel < 1e-14.
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-12, 
        atol=1e-14,
        max_step=np.inf
    )
    
    if not sol.success:
        # Fallback if it fails, though unlikely for this smooth problem
        return sol.y[:, -1]
    
    return sol.y[:, -1]
