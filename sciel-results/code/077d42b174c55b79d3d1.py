"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """E. coli growth rate from population density, substrate, temperature and pH."""
    # Normalize inputs to improve conditioning for BFGS
    # Based on training ranges:
    # b: [1, 4.998] -> center 3, scale 1
    # s: [1, 4.999] -> center 3, scale 1
    # temp: [1.002, 39.98] -> center 20, scale 10
    # pH: [2, 12] -> center 7, scale 2
    
    b_norm = (b - 3.0) / 1.0
    s_norm = (s - 3.0) / 1.0
    temp_norm = (temp - 20.0) / 10.0
    ph_norm = (pH - 7.0) / 2.0
    
    # Unpack parameters
    w1 = params[0]
    w2 = params[1]
    w3 = params[2]
    w4 = params[3]
    w5 = params[4]
    w6 = params[5]
    w7 = params[6]
    w8 = params[7]
    w9 = params[8]
    w10 = params[9]
    
    # Construct features
    # Linear terms
    f1 = s_norm
    f2 = temp_norm
    f3 = ph_norm
    f4 = b_norm
    
    # Quadratic terms for curvature (temperature and pH bell shapes)
    f5 = temp_norm ** 2
    f6 = ph_norm ** 2
    
    # Interaction terms
    f7 = s_norm * temp_norm
    f8 = s_norm * ph_norm
    f9 = temp_norm * ph_norm
    
    # Combine
    y = (w1 * f1 + 
         w2 * f2 + 
         w3 * f3 + 
         w4 * f4 + 
         w5 * f5 + 
         w6 * f6 + 
         w7 * f7 + 
         w8 * f8 + 
         w9 * f9 + 
         w10)
         
    return y
# EVOLVE-BLOCK-END
