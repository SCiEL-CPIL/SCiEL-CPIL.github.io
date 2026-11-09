"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a centered and scaled polynomial basis.
    Centering the inputs (subtracting mean) and scaling (dividing by range) helps
    decorrelate the polynomial terms and keeps parameter magnitudes O(1),
    improving BFGS convergence.
    
    params: 
    0: Bias
    1: Linear x
    2: Linear v
    3: Quadratic x
    4: Quadratic v
    5: Cross x*v
    6: Cubic x
    7: Cubic v
    8: x^2 v
    9: x v^2
    """
    # Deadline guard (safety margin)
    start_time = time.time()
    
    # Convert inputs to numpy arrays if not already
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64).flatten()
    
    # Center and Scale inputs
    # Estimated means from ranges:
    # x: [-0.5728, 0.5275] -> mean ~ -0.02, range ~ 1.1
    # v: [-0.1754, 0.2821] -> mean ~ 0.05, range ~ 0.46
    x_mean = -0.02
    v_mean = 0.05
    
    x_scale = 0.6
    v_scale = 0.3
    
    x_s = (x - x_mean) / x_scale
    v_s = (v - v_mean) / v_scale
    
    # Polynomial Basis
    T1_x = x_s
    T1_v = v_s
    T2_x = x_s * x_s
    T2_v = v_s * v_s
    T3_x = x_s * x_s * x_s
    T3_v = v_s * v_s * v_s
    
    # Cross terms
    T1_x_T1_v = x_s * v_s
    T2_x_T1_v = x_s * x_s * v_s
    T1_x_T2_v = x_s * v_s * v_s
    
    # Construct the linear combination
    pred = (params[0] + 
            params[1] * T1_x + 
            params[2] * T1_v + 
            params[3] * T2_x + 
            params[4] * T2_v + 
            params[5] * T1_x_T1_v + 
            params[6] * T3_x + 
            params[7] * T3_v + 
            params[8] * T2_x_T1_v + 
            params[9] * T1_x_T2_v)
            
    # Ensure output is finite
    if not np.all(np.isfinite(pred)):
        return np.zeros_like(x)
        
    # Check time (should be negligible, but good practice)
    if time.time() - start_time > 140:
         return pred

    return pred
# EVOLVE-BLOCK-END
