"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate from population density, substrate, temperature and pH.
    
    Uses a sparse sum of four Gaussian basis functions in 3D space (temp, pH, s),
    with a global linear term for b.
    
    Parameters:
    p[0]: amplitude for Gaussian 1
    p[1]: amplitude for Gaussian 2
    p[2]: amplitude for Gaussian 3
    p[3]: amplitude for Gaussian 4
    p[4]: shared isotropic width (sigma)
    p[5]: center temp for Gaussian 1
    p[6]: center pH for Gaussian 1
    p[7]: center s for Gaussian 1
    p[8]: center temp for Gaussian 2
    p[9]: center pH for Gaussian 2
    """
    # Note: We have 10 params.
    # We need centers for 4 Gaussians in 3D. That's 12 floats just for centers.
    # Plus 4 amplitudes and 1 width = 17 params. Too many.
    
    # Revised plan:
    # Use 2 Gaussians instead of 4 to fit within 10 params.
    # Or use a shared center structure?
    
    # Let's try 2 Gaussians in 2D (temp, pH) since s and b might be monotonic or simpler.
    # Or 2 Gaussians in 3D but with fixed centers? No, centers need to be fitted.
    
    # Let's try:
    # 1 Gaussian in 3D (temp, pH, s) + 1 Gaussian in 3D (temp, pH, s)
    # Params:
    # p[0]: amp1
    # p[1]: amp2
    # p[2]: sigma (shared)
    # p[3]: center1_temp
    # p[4]: center1_pH
    # p[5]: center1_s
    # p[6]: center2_temp
    # p[7]: center2_pH
    # p[8]: center2_s
    # p[9]: bias/intercept
    
    # This fits exactly 10 params.
    # b is ignored? Or included in a simple linear term?
    # If we include b linearly, we need another param.
    # Maybe drop one center? Or assume s is constant?
    # Let's assume b has a weak effect or is captured by the intercept if we treat it as a constant offset?
    # Better: Use 1 Gaussian in 2D (temp, pH) and a linear term for b and s?
    
    # Alternative:
    # p[0]: intercept
    # p[1]: coeff for b
    # p[2]: coeff for s
    # p[3]: coeff for temp
    # p[4]: coeff for pH
    # p[5]: amp for Gaussian (temp, pH)
    # p[6]: center_temp
    # p[7]: center_pH
    # p[8]: sigma
    # p[9]: coeff for b*s interaction?
    
    # This is a hybrid: Linear + 1 Gaussian bump.
    # This allows capturing non-monotonic behavior in temp/pH while handling b/s linearly.
    
    p = params
    
    # Normalize inputs for better conditioning of the Gaussian
    # temp: [1, 40] -> center 20, scale 20
    # pH: [2, 12] -> center 7, scale 5
    # s: [1, 5] -> center 3, scale 2
    # b: [1, 5] -> center 3, scale 2
    
    t_norm = (temp - 20.0) / 20.0
    ph_norm = (pH - 7.0) / 5.0
    s_norm = (s - 3.0) / 2.0
    b_norm = (b - 3.0) / 2.0
    
    # Linear terms
    linear = p[0] + p[1]*b_norm + p[2]*s_norm + p[3]*t_norm + p[4]*ph_norm
    
    # Gaussian term in (temp, pH) space
    # dist_sq = (t_norm - center_t)^2 + (ph_norm - center_p)^2
    center_t = p[6]
    center_p = p[7]
    sigma = p[8] + 1e-6  # ensure positive
    
    dist_sq = (t_norm - center_t)**2 + (ph_norm - center_p)**2
    gaussian = p[5] * np.exp(-dist_sq / (2 * sigma**2))
    
    # Interaction term
    interaction = p[9] * (b_norm * s_norm)
    
    return linear + gaussian + interaction
# EVOLVE-BLOCK-END
