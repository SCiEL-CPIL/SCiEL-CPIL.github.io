"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.integrate

def integrate(f, t_span, y0):
    """
    Main integration function using Richardson Extrapolation with DOP853.
    Strategy: Run DOP853 with step size H, then with H/2.
    Combine results to get a 9th-order accurate solution.
    This trades extra function calls for significantly higher accuracy.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    deadline = time.time() + 300.0
    
    # Initial step size for the "coarse" run
    # We want enough steps to be in the asymptotic regime but not too many.
    # 100 points suggests a spatial scale. Let's start with 200 steps for the coarse run.
    num_steps_coarse = 200
    max_step_coarse = (t1 - t0) / num_steps_coarse
    
    # Run 1: Coarse
    result1 = scipy.integrate.solve_ivp(
        f, t_span, y0,
        method='DOP853',
        rtol=1e-12,
        atol=1e-14,
        max_step=max_step_coarse
    )
    
    if not result1.success:
        return result1.y[:, -1] if result1.y.size > 0 else y0
    
    y_coarse = result1.y[:, -1]
    calls_coarse = result1.nfev
    
    # Check deadline
    if time.time() > deadline - 10.0:
        return y_coarse
        
    # Run 2: Fine (half step size)
    max_step_fine = max_step_coarse / 2.0
    
    result2 = scipy.integrate.solve_ivp(
        f, t_span, y0,
        method='DOP853',
        rtol=1e-12,
        atol=1e-14,
        max_step=max_step_fine
    )
    
    if not result2.success:
        return y_coarse
        
    y_fine = result2.y[:, -1]
    calls_fine = result2.nfev
    
    total_calls = calls_coarse + calls_fine
    
    # Check deadline
    if time.time() > deadline - 5.0:
        return y_fine # Fine is usually better than coarse
        
    # Richardson Extrapolation
    # DOP853 is 8th order.
    # y(h) = y_true + C * h^8 + O(h^9)
    # y(h/2) = y_true + C * (h/2)^8 + O(h^9)
    # 2^8 * y(h/2) - y(h) = (2^8 - 1) * y_true + O(h^9)
    # y_extrap = (2^8 * y_fine - y_coarse) / (2^8 - 1)
    
    order = 8
    factor = (2.0 ** order)
    y_extrap = (factor * y_fine - y_coarse) / (factor - 1.0)
    
    return y_extrap
# EVOLVE-BLOCK-END
