"""Rewritten program with a centered cubic polynomial basis for nonlinear oscillator regression."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a centered cubic polynomial basis.
    
    Inputs are centered and scaled to improve conditioning for the optimizer.
    Basis terms:
    1. 1
    2. x_c
    3. v_c
    4. x_c^2
    5. v_c^2
    6. x_c * v_c
    7. x_c^3
    8. v_c^3
    9. x_c^2 * v_c
    10. x_c * v_c^2
    """
    # Estimated centers and scales based on training ranges provided in the prompt
    # x range: [-0.5728, 0.5275] -> mean ~ -0.0226, half-range ~ 0.55
    # v range: [-0.1754, 0.2821] -> mean ~ 0.0533, half-range ~ 0.23
    
    x_mean = -0.0226
    v_mean = 0.0533
    x_scale = 0.55
    v_scale = 0.23
    
    x_c = (x - x_mean) / x_scale
    v_c = (v - v_mean) / v_scale
    
    # Construct basis
    t1 = np.ones_like(x_c)
    t2 = x_c
    t3 = v_c
    t4 = x_c * x_c
    t5 = v_c * v_c
    t6 = x_c * v_c
    t7 = x_c * x_c * x_c
    t8 = v_c * v_c * v_c
    t9 = x_c * x_c * v_c
    t10 = x_c * v_c * v_c
    
    # Linear combination
    y = (params[0] * t1 + 
         params[1] * t2 + 
         params[2] * t3 + 
         params[3] * t4 + 
         params[4] * t5 + 
         params[5] * t6 + 
         params[6] * t7 + 
         params[7] * t8 + 
         params[8] * t9 + 
         params[9] * t10)
         
    return y
# EVOLVE-BLOCK-END
