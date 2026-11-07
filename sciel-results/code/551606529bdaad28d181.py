import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Mechanistic Kinetic Model for E. coli growth rate.
    Uses Monod kinetics for substrates (s, b) and Gaussian stress factors
    for temperature and pH, with linear corrections.
    
    Parameters:
    p0: Base growth rate scale
    p1: ln(K_s) - Monod constant for substrate s
    p2: ln(K_b) - Monod constant for substrate b
    p3: ln(1/sigma_T^2) - Temperature sensitivity (inverse variance)
    p4: T_opt - Optimal temperature (normalized)
    p5: ln(1/sigma_pH^2) - pH sensitivity (inverse variance)
    p6: pH_opt - Optimal pH (normalized)
    p7: Linear temperature correction
    p8: Linear pH correction
    p9: Final scaling factor
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs for stability
    # b in [1, 5] -> center 3, scale 2
    b_norm = (b - 3.0) / 2.0
    # s in [1, 5] -> center 3, scale 2
    s_norm = (s - 3.0) / 2.0
    # temp in [1, 40] -> center 20.5, scale 19.5
    t_norm = (temp - 20.5) / 19.5
    # pH in [2, 12] -> center 7, scale 5
    ph_norm = (pH - 7.0) / 5.0
    
    # Monod terms using raw inputs for physical K values
    # Ensure K is positive by using exp
    K_s = np.exp(p1)
    K_b = np.exp(p2)
    
    # Avoid division by zero or negative K (exp is always positive)
    monod_s = s / (K_s + s)
    monod_b = b / (K_b + b)
    
    # Gaussian stress factors
    # Ensure variance inverse is positive
    inv_var_T = np.exp(p3)
    inv_var_pH = np.exp(p5)
    
    # Temperature factor: Gaussian peak + linear shift
    temp_factor = np.exp(-inv_var_T * (t_norm - p4)**2 + p7 * t_norm)
    
    # pH factor: Gaussian peak + linear shift
    ph_factor = np.exp(-inv_var_pH * (ph_norm - p6)**2 + p8 * ph_norm)
    
    # Combine
    pred = p0 * p9 * monod_s * monod_b * temp_factor * ph_factor
    
    return pred
