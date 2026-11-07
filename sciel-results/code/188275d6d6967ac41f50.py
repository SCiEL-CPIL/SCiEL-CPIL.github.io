# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale input to [-1, 1] for Chebyshev polynomials
    u = x / 3.0
    
    # Clamp to [-1, 1] to avoid numerical issues
    u_clamped = np.clip(u, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_0 to T_5
    # We will only use up to degree 5 to reduce overfitting risk
    # and improve conditioning for BFGS.
    
    t0 = np.ones_like(u_clamped)
    t1 = u_clamped.copy()
    
    result = params[0] * t0 + params[1] * t1
    
    # Calculate higher terms only up to degree 5
    if len(params) > 2:
        t_prev2 = t0
        t_prev1 = t1
        for k in range(2, 6):  # Calculate T_2 to T_5
            # T_k = 2*u*T_{k-1} - T_{k-2}
            t_curr = 2.0 * u_clamped * t_prev1 - t_prev2
            result += params[k] * t_curr
            t_prev2 = t_prev1
            t_prev1 = t_curr
            
    # Note: params[6] to params[9] are ignored in the prediction.
    # They will remain at their initial value (1.0) or drift, but since
    # they don't affect the output, they don't impact the MSE directly.
    # This effectively constrains the model to a degree-5 polynomial.
    
    return result
# EVOLVE-BLOCK-END
