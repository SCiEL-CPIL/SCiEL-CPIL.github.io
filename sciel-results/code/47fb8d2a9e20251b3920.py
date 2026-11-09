"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a Generalized Additive Model (GAM).
    
    Inputs are assumed to be in the following ranges based on training data:
    - b: [1, 5]
    - s: [1, 5]
    - temp: [1, 40]
    - pH: [2, 12]
    
    We use scaled polynomial basis functions for each variable to ensure
    numerical stability and convexity for the BFGS optimizer.
    
    Parameters:
    - params[0]: Bias
    - params[1], params[2]: Linear, Quadratic basis for b
    - params[3], params[4]: Linear, Quadratic basis for s
    - params[5], params[6]: Linear, Quadratic basis for temp
    - params[7], params[8]: Linear, Quadratic basis for pH
    - params[9]: Cubic basis for temp (temperature sensitivity)
    """
    # Scale inputs to approximately [-1, 1] or similar centered range
    # b: center 3, scale 2
    b_scaled = (b - 3.0) / 2.0
    # s: center 3, scale 2
    s_scaled = (s - 3.0) / 2.0
    # temp: center 20, scale 10
    temp_scaled = (temp - 20.0) / 10.0
    # pH: center 7, scale 5
    ph_scaled = (pH - 7.0) / 5.0
    
    # Basis functions
    # For each variable, we use linear and quadratic terms.
    # For temp, we add a cubic term as the 10th parameter.
    
    b_lin = b_scaled
    b_quad = b_scaled ** 2
    
    s_lin = s_scaled
    s_quad = s_scaled ** 2
    
    temp_lin = temp_scaled
    temp_quad = temp_scaled ** 2
    temp_cub = temp_scaled ** 3
    
    ph_lin = ph_scaled
    ph_quad = ph_scaled ** 2
    
    # Combine linearly
    pred = (params[0] + 
            params[1] * b_lin + params[2] * b_quad +
            params[3] * s_lin + params[4] * s_quad +
            params[5] * temp_lin + params[6] * temp_quad +
            params[7] * ph_lin + params[8] * ph_quad +
            params[9] * temp_cub)
            
    return pred
# EVOLVE-BLOCK-END
