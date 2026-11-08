"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Strategy: Empirical additive model with tanh saturation.
    - Linear in parameters (convex optimization landscape for BFGS)
    - Includes tanh(strain) to model plastic saturation/asymptotic stress
    - Includes interaction terms between strain and temperature
    - Polynomial terms up to 2nd/3rd order for curvature
    
    Parameters (10 total):
    p[0]: Intercept (offset)
    p[1]: Linear strain coefficient (Young's modulus approx)
    p[2]: Quadratic strain coefficient (hardening/softening)
    p[3]: Cubic strain coefficient
    p[4]: Tanh saturation coefficient (plastic limit)
    p[5]: Linear temp coefficient
    p[6]: Quadratic temp coefficient
    p[7]: Strain-Temp interaction (linear-linear)
    p[8]: Strain^2-Temp interaction
    p[9]: Tanh-Temp interaction
    """
    # Extract parameters
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[6] # Wait, p[7] is index 7
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]

    # Precompute basis functions
    # Center temperature for stability
    T_c = temp - 0.5
    
    # Strain powers
    s1 = strain
    s2 = strain * strain
    s3 = s2 * strain
    
    # Saturation term
    s_tanh = np.tanh(strain)
    
    # Temp powers
    t1 = T_c
    t2 = T_c * T_c
    
    # Interactions
    st1 = s1 * t1
    s2t1 = s2 * t1
    st_tanh = s_tanh * t1

    # Construct prediction
    # Additive model:
    # Stress = p0 + p1*s + p2*s^2 + p3*s^3 + p4*tanh(s) 
    #          + p5*t + p6*t^2 
    #          + p7*s*t + p8*s^2*t + p9*tanh(s)*t
    
    pred = (p0 
            + p1 * s1 
            + p2 * s2 
            + p3 * s3 
            + p4 * s_tanh 
            + p5 * t1 
            + p6 * t2 
            + p7 * st1 
            + p8 * s2t1 
            + p9 * st_tanh)
            
    return pred
# EVOLVE-BLOCK-END
