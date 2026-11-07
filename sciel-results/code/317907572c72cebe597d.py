import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [0, 1] based on training ranges
    # b: [1, 5] -> [0, 1]
    # s: [1, 5] -> [0, 1]
    # temp: [1, 40] -> [0, 1]
    # pH: [2, 12] -> [0, 1]
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    t_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    p = params
    
    # 1. Substrate: Monod-like saturation
    # p[0] controls half-saturation. Squared to ensure positivity.
    k_s = p[0]**2 + 1e-6
    term_s = s_norm / (s_norm + k_s)
    
    # 2. Temperature: Gaussian bell curve
    # p[1] center, p[2] width (squared for positivity)
    t_center = p[1]
    t_width = p[2]**2 + 1e-6
    term_t = np.exp(-((t_norm - t_center)**2) / (2 * t_width))
    
    # 3. pH: Gaussian bell curve
    # p[3] center, p[4] width (squared for positivity)
    ph_center = p[3]
    ph_width = p[4]**2 + 1e-6
    term_ph = np.exp(-((ph_norm - ph_center)**2) / (2 * ph_width))
    
    # 4. Biomass: Logistic density-dependent inhibition
    # Growth rate decreases as biomass increases.
    # Model: term_b = p[7] / (1 + exp(p[5] * (b_norm - p[6])))
    # p[5] is the steepness of the inhibition.
    # p[6] is the midpoint of the inhibition (biomass level where effect is half-max).
    # p[7] is the scaling factor for the biomass term.
    
    # Clip the exponent to prevent overflow/underflow issues with BFGS
    arg = p[5] * (b_norm - p[6])
    arg = np.clip(arg, -50, 50)
    term_b = p[7] / (1.0 + np.exp(arg))
    
    # 5. Combined Model
    # Multiplicative interaction of environmental factors and density.
    # p[8] is an additive offset (e.g., background death rate or noise).
    # p[9] is a global scaling factor.
    
    y = p[9] * term_s * term_t * term_ph * term_b + p[8]
    
    return y
