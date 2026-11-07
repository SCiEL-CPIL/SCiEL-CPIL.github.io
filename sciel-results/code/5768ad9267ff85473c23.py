"""E. coli growth rate model with Monod kinetics, piecewise-linear temp/pH plateaus, and density inhibition."""
import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model with hard-constrained temperature/pH response.
    
    Structure:
    - Substrate: Monod kinetics s / (Ks + s)
    - Temperature: piecewise-linear plateau with hard zero-growth boundaries
    - pH: piecewise-linear plateau with hard zero-growth boundaries  
    - Density: Haldane-type inhibition (b/Kb)/(1+(b/Kb)^2)
    - Overall: mu_max * f(s) * f_T(temp) * f_pH(pH) * f_b(b) + offset
    
    Parameters (10 total):
    p0: mu_max (max growth rate)
    p1: Ks (half-saturation for substrate)
    p2: T_lower (lower temp boundary where growth becomes zero)
    p3: T_upper (upper temp boundary where growth becomes zero)
    p4: pH_lower (lower pH boundary where growth becomes zero)
    p5: pH_upper (upper pH boundary where growth becomes zero)
    p6: Kb (density half-inhibition)
    p7: T_plateau_width (width of optimal temp plateau)
    p8: offset (baseline)
    p9: additional scaling
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure positive values for widths and half-saturation constants
    Ks = np.exp(p1) + 1e-6
    Kb = np.exp(p6) + 1e-6
    
    # Temperature boundaries: T_lower < T_opt - plateau_width, T_upper > T_opt + plateau_width
    # Use transformations to enforce ordering and positivity
    T_lower = 10.0 + np.exp(p2) * 0.5  # in [10, ~15]
    T_upper = 40.0 - np.exp(p3) * 0.5  # in [~35, 40]
    
    # pH boundaries: pH_lower < pH_opt, pH_upper > pH_opt
    pH_lower = 2.0 + np.exp(p4) * 0.5  # in [2, ~3]
    pH_upper = 12.0 - np.exp(p5) * 0.5  # in [~11, 12]
    
    # Plateau width for temperature
    T_plateau_half_width = np.exp(p7) * 2.0 + 1.0  # in [1, ~3]
    
    # Temperature response: piecewise linear
    # Zero below T_lower, linear rise to 1.0 at T_opt - T_plateau_half_width
    # Plateau at 1.0 from T_opt - T_plateau_half_width to T_opt + T_plateau_half_width
    # Linear fall to 0.0 at T_upper, zero above T_upper
    T_opt = (T_lower + T_upper) / 2.0
    
    # Compute temperature factor
    f_T = np.zeros_like(temp, dtype=np.float64)
    
    # Rising edge: from T_lower to T_opt - T_plateau_half_width
    T_rise_start = T_lower
    T_rise_end = T_opt - T_plateau_half_width
    mask_rise = (temp >= T_rise_start) & (temp < T_rise_end)
    if np.any(mask_rise):
        slope_rise = 1.0 / max(T_rise_end - T_rise_start, 1e-6)
        f_T[mask_rise] = (temp[mask_rise] - T_rise_start) * slope_rise
    
    # Plateau: from T_opt - T_plateau_half_width to T_opt + T_plateau_half_width
    T_plateau_start = T_opt - T_plateau_half_width
    T_plateau_end = T_opt + T_plateau_half_width
    mask_plateau = (temp >= T_plateau_start) & (temp <= T_plateau_end)
    f_T[mask_plateau] = 1.0
    
    # Falling edge: from T_opt + T_plateau_half_width to T_upper
    T_fall_start = T_opt + T_plateau_half_width
    T_fall_end = T_upper
    mask_fall = (temp > T_fall_start) & (temp <= T_fall_end)
    if np.any(mask_fall):
        slope_fall = -1.0 / max(T_fall_end - T_fall_start, 1e-6)
        f_T[mask_fall] = (temp[mask_fall] - T_fall_end) * slope_fall
    
    # pH response: piecewise linear with similar structure
    pH_opt = (pH_lower + pH_upper) / 2.0
    pH_plateau_half_width = 1.5  # fixed reasonable width
    
    f_pH = np.zeros_like(pH, dtype=np.float64)
    
    # Rising edge
    pH_rise_start = pH_lower
    pH_rise_end = pH_opt - pH_plateau_half_width
    mask_pH_rise = (pH >= pH_rise_start) & (pH < pH_rise_end)
    if np.any(mask_pH_rise):
        slope_pH_rise = 1.0 / max(pH_rise_end - pH_rise_start, 1e-6)
        f_pH[mask_pH_rise] = (pH[mask_pH_rise] - pH_rise_start) * slope_pH_rise
    
    # Plateau
    pH_plateau_start = pH_opt - pH_plateau_half_width
    pH_plateau_end = pH_opt + pH_plateau_half_width
    mask_pH_plateau = (pH >= pH_plateau_start) & (pH <= pH_plateau_end)
    f_pH[mask_pH_plateau] = 1.0
    
    # Falling edge
    pH_fall_start = pH_opt + pH_plateau_half_width
    pH_fall_end = pH_upper
    mask_pH_fall = (pH > pH_fall_start) & (pH <= pH_fall_end)
    if np.any(mask_pH_fall):
        slope_pH_fall = -1.0 / max(pH_fall_end - pH_fall_start, 1e-6)
        f_pH[mask_pH_fall] = (pH[mask_pH_fall] - pH_fall_end) * slope_pH_fall
    
    # Monod substrate term
    monod_s = s / (Ks + s)
    
    # Density inhibition: Haldane-like
    b_ratio = b / Kb
    f_b = b_ratio / (1.0 + b_ratio * b_ratio)
    
    # Combine
    mu_max = np.exp(p0) + 1e-6
    scaling = np.exp(p9) + 1e-6
    
    result = mu_max * scaling * monod_s * f_T * f_pH * f_b + p8
    
    # Clip to reasonable range
    return np.clip(result, -1.0, 3.0)
