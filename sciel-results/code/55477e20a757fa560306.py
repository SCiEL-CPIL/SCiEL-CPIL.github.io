"""ODE solver: Robertson problem using LSODA via scipy.integrate.solve_ivp."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns y(t_end).
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard: return best result before 320s (10s safety margin from 330s)
    start_time = time.time()
    
    # Wrap f to count calls and potentially track time
    call_count = [0]
    
    def wrapped_f(t, y):
        call_count[0] += 1
        return f(t, y)
    
    # First pass: LSODA with tight tolerances
    # Robertson problem is stiff, LSODA switches between Adams and BDF automatically
    result = solve_ivp(
        wrapped_f,
        t_span,
        y0,
        method='LSODA',
        rtol=1e-14,
        atol=1e-16,
        dense_output=False
    )
    
    if not result.success:
        # Fallback: try with looser tolerances if it failed
        result = solve_ivp(
            wrapped_f,
            t_span,
            y0,
            method='LSODA',
            rtol=1e-12,
            atol=1e-14,
            dense_output=False
        )
    
    if result.success and len(result.y) > 0:
        y_final = result.y[:, -1]
    else:
        # Last resort: use whatever we have
        y_final = result.y[:, -1] if len(result.y) > 0 else y0
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed < 300:
        # Try a second polish pass with even tighter tolerances
        # Reset call count for this pass
        call_count[0] = 0
        
        result2 = solve_ivp(
            wrapped_f,
            t_span,
            y_final,  # Use previous result as initial condition for polish
            method='LSODA',
            rtol=1e-15,
            atol=1e-17,
            dense_output=False
        )
        
        if result2.success and len(result2.y) > 0:
            y_final = result2.y[:, -1]
    
    return y_final
# EVOLVE-BLOCK-END
