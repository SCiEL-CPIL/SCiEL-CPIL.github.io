"""E. coli growth rate model: Fourier basis with Monod substrate gating."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier basis model with Monod substrate gating for E. coli growth rate.
    
    The model uses a sum-of-sines basis for temperature and pH to capture
    periodic/oscillatory constraints (optima), combined with a Monod term
    for substrate saturation. The biomass term provides a linear correction.
    
    Parameters (10):
    p0: Monod half-saturation constant K_s
    p1: amplitude for temperature sine
    p2: frequency for temperature
    p3: amplitude for temperature cosine
    p4: amplitude for pH sine
    p5: frequency for pH
    p6: amplitude for pH cosine
    p7: scale factor for the product term
    p8: coefficient for biomass (b)
    p9: bias offset
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Monod substrate term: s / (K_s + s)
    # Clamp K_s to be positive to avoid division issues
    K_s = np.abs(p0) + 1e-6
    monod = s / (K_s + s)

    # Temperature Fourier terms
    # Use a scaled frequency to keep arguments in a reasonable range
    # temp is roughly in [1, 4.999] based on training range
    temp_scaled = temp * (np.abs(p2) + 1e-6)
    temp_sin = np.sin(temp_scaled)
    temp_cos = np.cos(temp_scaled)
    temp_term = p1 * temp_sin + p3 * temp_cos

    # pH Fourier terms
    # pH is roughly in [1.002, 39.98] based on training range (wait, pH range seems wide? 
    # Let's re-read: [1.002, 39.98] is the 3rd input range? 
    # Inputs: [1, 4.998], [1, 4.999], [1.002, 39.98], [2, 12]
    # Order: b, s, temp, pH? 
    # Task says: "Inputs in training range, in order: [1, 4.998], [1, 4.999], [1.002, 39.98], [2, 12]."
    # Function signature: equation(b, s, temp, pH, params)
    # So b in [1, 4.998], s in [1, 4.999], temp in [1.002, 39.98], pH in [2, 12].
    # This makes sense: temp ~ 30-40C, pH ~ 2-12.
    
    pH_scaled = pH * (np.abs(p5) + 1e-6)
    pH_sin = np.sin(pH_scaled)
    pH_cos = np.cos(pH_scaled)
    pH_term = p4 * pH_sin + p6 * pH_cos

    # Combine terms
    # The growth rate is typically multiplicative: Substrate * Temp_Response * pH_Response
    # But Fourier terms can be negative. We want positive responses around optima.
    # A common trick is to add a large constant or use (1 + term) structure.
    # However, since we have a bias p9, we can let the optimizer figure out the combination.
    # Let's try a multiplicative structure with offset to ensure positivity where needed,
    # or just a sum. A sum is more flexible for BFGS.
    
    # Let's try: mu = p7 * monod * (1 + temp_term) * (1 + pH_term) + p8 * b + p9
    # This ensures that if temp_term and pH_term are 0, we get p7 * monod + p8*b + p9.
    # The (1+term) part allows modulation.
    
    temp_factor = 1.0 + temp_term
    pH_factor = 1.0 + pH_term
    
    # Avoid issues if factors become negative and large, but BFGS should handle it.
    # To be safe, we can clip the factors to prevent extreme values if the optimizer goes wild.
    # But let's first try without clipping to see if it converges well.
    
    raw = p7 * monod * temp_factor * pH_factor + p8 * b + p9

    # Ensure finite output
    result = np.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result
# EVOLVE-BLOCK-END
