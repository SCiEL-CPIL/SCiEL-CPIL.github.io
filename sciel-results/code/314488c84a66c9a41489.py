import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model combining Monod kinetics with Gaussian environmental
    responses and density inhibition.
    
    Structure:
    mu = p0 * Monod(s) * Gaussian_T(temp) * Gaussian_pH(pH) * Inhibition(b) + p9
    
    Parameters (10 total):
    p0: Max growth rate scale
    p1: Substrate half-saturation Ks (positive via abs + small offset)
    p2: Temperature optimum center (offset from 37)
    p3: Temperature width sigma (positive via abs + small offset)
    p4: pH optimum center (offset from 7)
    p5: pH width sigma (positive via abs + small offset)
    p6: Density inhibition constant Kb (positive via abs + small offset)
    p7: Density inhibition strength
    p8: T-pH interaction correction coefficient
    p9: Bias/offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure positive constants for stability
    Ks = np.abs(p1) + 0.1
    sigma_T = np.abs(p3) + 1.0
    sigma_pH = np.abs(p5) + 1.0
    Kb = np.abs(p6) + 0.1
    
    # Centered variables for better conditioning
    T_opt = 37.0 + p2
    pH_opt = 7.0 + p4
    
    # Monod substrate term: S / (Ks + S)
    monod_s = s / (Ks + s)
    
    # Gaussian environmental terms
    T_centered = temp - T_opt
    pH_centered = pH - pH_opt
    
    gauss_T = np.exp(-(T_centered ** 2) / (2.0 * sigma_T ** 2))
    gauss_pH = np.exp(-(pH_centered ** 2) / (2.0 * sigma_pH ** 2))
    
    # Density inhibition: 1 - strength * (b / (Kb + b))
    # This goes from 1 (low density) to 1-strength (high density)
    inhibition = 1.0 - p7 * (b / (Kb + b))
    
    # Interaction term: small correction based on T and pH deviations
    # This allows the optima to shift slightly based on the other variable
    interaction = 1.0 + p8 * (T_centered * pH_centered) / (sigma_T * sigma_pH)
    
    # Combine
    growth = p0 * monod_s * gauss_T * gauss_pH * inhibition * interaction + p9
    
    return growth
