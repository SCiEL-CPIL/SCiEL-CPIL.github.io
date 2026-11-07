"""Symbolic regression: an equation for E. coli growth rate."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled with an additive basis expansion using
    biologically motivated non-linear features.
    
    Features:
    - Monod saturation for substrate
    - Gaussian temperature optimum
    - Gaussian pH optimum
    - Linear density term
    - Key interaction terms between these features
    
    This structure allows the optimizer to fit complex shapes (saturation, 
    optima) that are difficult to capture with simple polynomials.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # 1. Substrate: Monod saturation
    # Ks = p1. We ensure Ks is positive to avoid division by zero or negative saturation.
    # Using abs(p1) ensures the denominator is always positive if p1 != 0.
    # If p1 is very small, it approaches step function; if large, linear.
    Ks = np.maximum(np.abs(p1), 1e-6)
    monod_s = s / (Ks + s)

    # 2. Temperature: Gaussian optimum
    # T_opt = p2, sigma_T = p3
    # We use a safe sigma to prevent division by zero in the exponent.
    T_opt = p2
    sigma_T = np.maximum(np.abs(p3), 1e-3)
    # The exponent is -0.5 * ((temp - T_opt) / sigma_T)^2
    # This is numerically stable for typical ranges.
    temp_diff = (temp - T_opt) / sigma_T
    temp_factor = np.exp(-0.5 * temp_diff * temp_diff)

    # 3. pH: Gaussian optimum
    # pH_opt = p4, sigma_pH = p5
    pH_opt = p4
    sigma_pH = np.maximum(np.abs(p5), 1e-3)
    ph_diff = (pH - pH_opt) / sigma_pH
    ph_factor = np.exp(-0.5 * ph_diff * ph_diff)

    # 4. Density: Linear term
    # We keep b linear here because the range is small [1, 5] and 
    # interactions with other terms handle the complexity.
    b_term = b

    # Construct prediction as a linear combination of basis functions
    # p0: Bias
    # p1 is used for Ks, so we don't use it as a coefficient for monod_s directly?
    # Wait, the task says params are fitted. In the previous best (inspiration 2), 
    # p1 was used for Ks AND the coefficient? No, in inspiration 2, p1 was Ks, 
    # and the coefficient for monod_s was... actually, looking at inspiration 2 code:
    # result = p0 + p1 * monod_s + ...
    # But Ks = max(abs(p1), 1e-6). This means p1 controls both the shape (Ks) and the amplitude.
    # This is a strong coupling. It might be better to separate them if possible, 
    # but we only have 10 params.
    # Let's look at the current program's parameter usage.
    # If I use p1 for Ks, I need another param for the amplitude of monod_s.
    # However, the optimizer can scale the amplitude by adjusting other terms if the basis is rich enough?
    # Actually, if I use p1 for Ks, the term is just `monod_s` (coefficient 1). 
    # But I want a coefficient. 
    # Let's re-evaluate the parameter allocation.
    
    # Option A: 
    # p0: Bias
    # p1: Ks (for Monod)
    # p2: T_opt
    # p3: sigma_T
    # p4: pH_opt
    # p5: sigma_pH
    # p6: Coeff for monod_s
    # p7: Coeff for temp_factor
    # p8: Coeff for ph_factor
    # p9: Coeff for b
    # This leaves no room for interactions. Interactions were key in the good scores.
    
    # Option B (from Inspiration 2 which scored 1.0436):
    # It used p1 for Ks AND as the coefficient? 
    # Code: Ks = max(abs(p1), 1e-6); ... result = p0 + p1 * monod_s + ...
    # This is a bit weird. It couples the shape and scale.
    # However, it scored better than the polynomial approach.
    
    # Let's try a hybrid:
    # We have 10 params.
    # 3 params for Temp (Opt, Sigma) -> 2 params (Opt, Sigma)
    # 3 params for pH (Opt, Sigma) -> 2 params (Opt, Sigma)
    # 1 param for Substrate (Ks) -> 1 param
    # That's 5 params for the "shape" of the non-linear features.
    # We have 5 params left for coefficients/bias/interactions.
    
    # Let's allocate:
    # p0: Bias
    # p1: Ks (Monod)
    # p2: T_opt
    # p3: sigma_T
    # p4: pH_opt
    # p5: sigma_pH
    # p6: Coeff for monod_s
    # p7: Coeff for temp_factor
    # p8: Coeff for ph_factor
    # p9: Coeff for (monod_s * temp_factor) -- A key interaction
    
    # What about b? We lost b. 
    # The current program had b. The seed had b.
    # E. coli growth is density dependent.
    
    # Let's swap: 
    # Maybe we don't need pH_opt and sigma_pH separately if we assume a fixed width? 
    # No, flexibility is good.
    
    # Alternative Allocation:
    # p0: Bias
    # p1: Ks
    # p2: T_opt
    # p3: pH_opt
    # p4: Coeff for monod_s
    # p5: Coeff for temp_factor (using fixed sigma? No, sigma is important)
    
    # Let's look at the ranges. 
    # Temp: 1-40. pH: 2-12.
    # If we fix sigma_T and sigma_pH, we save 2 params.
    # Fixed sigma_T = 5.0 (approx 1/4 of range)
    # Fixed sigma_pH = 2.0 (approx 1/5 of range)
    # This frees up p3 and p5.
    
    # Revised Plan:
    # p0: Bias
    # p1: Ks (Monod)
    # p2: T_opt
    # p3: pH_opt
    # p4: Coeff for monod_s
    # p5: Coeff for temp_factor
    # p6: Coeff for ph_factor
    # p7: Coeff for b
    # p8: Coeff for (monod_s * temp_factor)
    # p9: Coeff for (temp_factor * ph_factor)
    
    # This allows us to model:
    # - Substrate saturation
    # - Temperature optimum
    # - pH optimum
    # - Density effect
    # - Substrate-Temp interaction
    # - Temp-pH interaction
    
    # This seems like a very strong basis.
    
    # Let's implement this.
    
    # Recalculate features with this parameter mapping
    Ks = np.maximum(np.abs(p1), 1e-6)
    monod_s = s / (Ks + s)
    
    T_opt = p2
    # Fixed sigma for temperature to save params
    sigma_T = 5.0
    temp_diff = (temp - T_opt) / sigma_T
    temp_factor = np.exp(-0.5 * temp_diff * temp_diff)
    
    pH_opt = p3
    # Fixed sigma for pH to save params
    sigma_pH = 2.0
    ph_diff = (pH - pH_opt) / sigma_pH
    ph_factor = np.exp(-0.5 * ph_diff * ph_diff)
    
    b_term = b
    
    pred = (
        p0
        + p4 * monod_s
        + p5 * temp_factor
        + p6 * ph_factor
        + p7 * b_term
        + p8 * monod_s * temp_factor
        + p9 * temp_factor * ph_factor
    )
    
    return np.nan_to_num(pred, nan=0.0, posinf=100.0, neginf=-100.0)
# EVOLVE-BLOCK-END
