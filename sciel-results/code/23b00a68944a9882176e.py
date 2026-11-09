# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model combining Monod kinetics with a flexible
    polynomial and interaction basis for temperature, pH, and population density.

    This hybrid approach uses the biologically-motivated Monod term for
    substrate saturation and a quadratic polynomial basis with key interactions
    for the other variables. The quadratic terms capture curvature (optimal
    ranges) for temperature and pH, while interaction terms (temp*pH, s*temp)
    capture synergistic or antagonistic effects. The model is linear in the
    parameters (except p0 for Ks), providing a stable optimization landscape
    for BFGS.

    Inputs:
    b: Population density (range ~[1, 5])
    s: Substrate concentration (range ~[1, 40])
    temp: Temperature (range ~[1, 40])
    pH: pH level (range ~[1, 14])

    Parameters (10):
    p0: Half-saturation constant Ks (for Monod term, used as exp(p0) for stability)
    p1: Scaling factor for the overall growth rate (multiplies Monod)
    p2: Linear coefficient for b
    p3: Linear coefficient for temp
    p4: Linear coefficient for pH
    p5: Quadratic coefficient for temp
    p6: Quadratic coefficient for pH
    p7: Interaction coefficient for temp * pH
    p8: Interaction coefficient for s * temp
    p9: Bias term
    """
    p = params

    # Normalize inputs to [0, 1] range for better conditioning
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 39.0
    temp_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 1.0) / 13.0

    # Monod term for substrate
    # Use exp(p0) to ensure Ks is positive
    Ks = np.exp(p[0])
    monod = s / (Ks + s)

    # Polynomial and interaction features
    # Linear terms
    b_lin = b_norm
    temp_lin = temp_norm
    ph_lin = ph_norm
    
    # Quadratic terms (curvature)
    temp_quad = temp_norm * temp_norm
    ph_quad = ph_norm * ph_norm

    # Interaction terms
    temp_ph_int = temp_norm * ph_norm
    s_temp_int = s_norm * temp_norm

    # Combine: 
    # Base growth is driven by Monod (substrate) scaled by p1.
    # Additive corrections from population, temperature, pH, and their interactions.
    # This structure allows the model to capture:
    # - Substrate saturation (Monod)
    # - Population density effect (linear)
    # - Temperature and pH optimal ranges (linear + quadratic)
    # - Temp-pH synergy/antagonism (interaction)
    # - Substrate-temperature coupling (interaction)
    
    base_growth = p[1] * monod
    
    # Additive corrections
    correction = (p[2] * b_lin 
                  + p[3] * temp_lin 
                  + p[4] * ph_lin 
                  + p[5] * temp_quad 
                  + p[6] * ph_quad 
                  + p[7] * temp_ph_int 
                  + p[8] * s_temp_int)
    
    result = base_growth + correction + p[9]

    # Ensure the output is finite
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
