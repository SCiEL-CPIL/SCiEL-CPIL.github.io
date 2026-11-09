"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a quadratic response surface with key interactions.
    
    Inputs:
    b: Population density [1, 5]
    s: Substrate concentration [1, 5]
    temp: Temperature [1, 40]
    pH: pH level [2, 12]
    
    Model:
    y = p0 + p1*b_c + p2*s_c + p3*t_c + p4*p_c 
        + p5*(s_c*t_c) + p6*(b_c*s_c) 
        + p7*(s_c^2) + p8*(t_c^2) + p9*(p_c^2)
        
    Centers chosen to normalize inputs for BFGS stability:
    b_c = b - 3.0
    s_c = s - 3.0
    t_c = temp - 20.0
    p_c = pH - 7.0
    """
    # Normalize inputs to improve optimization conditioning
    b_c = b - 3.0
    s_c = s - 3.0
    t_c = temp - 20.0
    p_c = pH - 7.0
    
    p = params
    
    # Pre-compute terms
    # Linear terms
    term_b = p[1] * b_c
    term_s = p[2] * s_c
    term_t = p[3] * t_c
    term_p = p[4] * p_c
    
    # Interaction terms (Substrate-Temp, Density-Substrate)
    term_st = p[5] * (s_c * t_c)
    term_bs = p[6] * (b_c * s_c)
    
    # Quadratic terms (Substrate, Temp, pH)
    term_s2 = p[7] * (s_c * s_c)
    term_t2 = p[8] * (t_c * t_c)
    term_p2 = p[9] * (p_c * p_c)
    
    # Intercept
    term_0 = p[0]
    
    # Sum all terms
    return term_0 + term_b + term_s + term_t + term_p + term_st + term_bs + term_s2 + term_t2 + term_p2
# EVOLVE-BLOCK-END
