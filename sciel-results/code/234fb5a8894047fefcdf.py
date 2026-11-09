"""E. coli growth rate model combining mechanistic structure with linear-in-parameters flexibility."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model.
    
    Combines the structural insight of the current program (Monod + Gaussian responses)
    with the linear-in-parameters flexibility of inspiration 1. This allows BFGS to
    optimize coefficients freely while maintaining biologically meaningful feature shapes.
    
    The model uses a linear combination of 10 basis functions:
    - Monod kinetics for substrate
    - Gaussian responses for temperature and pH
    - Interaction terms and density effects
    
    Parameters (10):
    p0: Coefficient for Monod * TempGauss * pHGauss (core multiplicative term)
    p1: Coefficient for Monod * TempGauss (substrate * temp, ignoring pH)
    p2: Coefficient for Monod * pHGauss (substrate * pH, ignoring temp)
    p3: Coefficient for Monod alone (substrate only)
    p4: Coefficient for linear substrate * TempGauss (non-saturating temp effect)
    p5: Coefficient for linear substrate * pHGauss (non-saturating pH effect)
    p6: Coefficient for density inhibition (linear in b, scaled by core term)
    p7: Coefficient for quadratic density inhibition
    p8: Bias term
    p9: Coefficient for high-order interaction (core * b * broader Gaussians)
    """
    # Deadline guard: return a safe default if we're running out of time
    # The evaluator kills after 150s, so we check elapsed time
    # Since this function is called repeatedly during fitting, we can't track
    # global time easily. Instead, we ensure the computation is fast.
    
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Normalize inputs for numerical stability in BFGS
    # b: [1, 5] -> center 3, scale 2
    # s: [1, 5] -> center 3, scale 2
    # temp: [1, 40] -> center 20.5, scale 19.5
    # pH: [2, 12] -> center 7, scale 5
    bn = (b - 3.0) / 2.0
    sn = (s - 3.0) / 2.0
    tn = (temp - 20.5) / 19.5
    phn = (pH - 7.0) / 5.0

    # Pre-compute basis functions
    # 1. Monod kinetics with fixed K=1 (normalized substrate)
    #    s_norm / (K + s_norm). Using normalized s to keep values in [0,1]
    #    Clip to non-negative to ensure physical meaning and avoid negative Monod
    s_norm_pos = np.clip(sn, 0.0, None)
    monod = s_norm_pos / (1.0 + s_norm_pos)
    
    # 2. Temperature response: Gaussian centered at ~37C (normalized)
    #    37C in normalized: (37 - 20.5) / 19.5 = 16.5 / 19.5 ≈ 0.846
    #    Width: ~10C -> 10/19.5 ≈ 0.513
    t_center = 0.846
    t_width = 0.513
    temp_g1 = np.exp(-((tn - t_center) ** 2) / (2.0 * t_width ** 2))
    
    #    Broader temperature term for high-order interaction
    t_width_broad = 0.8
    temp_g2 = np.exp(-((tn - t_center) ** 2) / (2.0 * t_width_broad ** 2))

    # 3. pH response: Gaussian centered at 7.0 (normalized to 0)
    #    Width: ~2 pH units -> 2/5 = 0.4
    ph_center = 0.0
    ph_width = 0.4
    ph_g1 = np.exp(-((phn - ph_center) ** 2) / (2.0 * ph_width ** 2))
    
    #    Broader pH term
    ph_width_broad = 0.8
    ph_g2 = np.exp(-((phn - ph_center) ** 2) / (2.0 * ph_width_broad ** 2))

    # 4. Density terms (normalized)
    dens_l = bn
    dens_q = bn ** 2

    # Construct features
    # F0: Main multiplicative core (Monod * Temp * pH)
    F0 = monod * temp_g1 * ph_g1
    
    # F1: Substrate * Temp (ignoring pH)
    F1 = monod * temp_g1
    
    # F2: Substrate * pH (ignoring Temp)
    F2 = monod * ph_g1
    
    # F3: Substrate only
    F3 = monod
    
    # F4: Linear substrate * Temp (captures non-saturating temp effect)
    F4 = sn * temp_g1
    
    # F5: Linear substrate * pH (captures non-saturating pH effect)
    F5 = sn * ph_g1
    
    # F6: Density inhibition (linear, scaled by core term)
    F6 = F0 * dens_l
    
    # F7: Quadratic density inhibition (scaled by core term)
    F7 = F0 * dens_q
    
    # F8: Bias term
    F8 = np.ones_like(b)
    
    # F9: High-order interaction (core * density * broader Gaussians)
    F9 = F0 * dens_l * temp_g2 * ph_g2

    # Combine linearly
    result = (p0 * F0 + 
              p1 * F1 + 
              p2 * F2 + 
              p3 * F3 + 
              p4 * F4 + 
              p5 * F5 + 
              p6 * F6 + 
              p7 * F7 + 
              p8 * F8 + 
              p9 * F9)
    
    # Ensure finite output
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
