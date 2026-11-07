import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [0, 1] based on training ranges
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    t_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    p = params
    
    # Sparse sum of four Gaussian basis functions
    # Each Gaussian is centered in 3D space (temp, pH, substrate)
    # Parameters:
    #   p[0:3] = center 1 (t, ph, s)
    #   p[3:6] = center 2 (t, ph, s)
    #   p[6:9] = center 3 (t, ph, s)
    #   p[9]   = shared isotropic width (sigma)
    
    # We use 3 Gaussians to fit within 10 params:
    # 3 centers * 3 dims = 9 params, 1 width = 10 params total
    # Wait, the directive says "four Gaussian basis functions". 
    # But 4 * 3 = 12 params for centers + 1 width = 13 params. Too many.
    # Let's stick to 3 Gaussians as per the "up to 10 fitted constants" constraint.
    # Or, we can use 2 Gaussians (2*3=6) + 1 width + 1 offset + 1 linear term?
    # Let's try 3 Gaussians as it fits exactly 10 params (9 centers + 1 width).
    
    c1_t, c1_ph, c1_s = p[0], p[1], p[2]
    c2_t, c2_ph, c2_s = p[3], p[4], p[5]
    c3_t, c3_ph, c3_s = p[6], p[7], p[8]
    sigma = p[9]
    
    # Ensure sigma is positive and not too small to avoid numerical issues
    sigma = np.maximum(sigma, 0.1)
    
    # Calculate squared distances for each Gaussian
    # d1^2 = (t - c1_t)^2 + (ph - c1_ph)^2 + (s - c1_s)^2
    d1_sq = (t_norm - c1_t)**2 + (ph_norm - c1_ph)**2 + (s_norm - c1_s)**2
    d2_sq = (t_norm - c2_t)**2 + (ph_norm - c2_ph)**2 + (s_norm - c2_s)**2
    d3_sq = (t_norm - c3_t)**2 + (ph_norm - c3_ph)**2 + (s_norm - c3_s)**2
    
    # Gaussian functions
    g1 = np.exp(-d1_sq / (2 * sigma**2))
    g2 = np.exp(-d2_sq / (2 * sigma**2))
    g3 = np.exp(-d3_sq / (2 * sigma**2))
    
    # Sum of Gaussians
    # Note: We don't have separate weights for each Gaussian because we only have 10 params.
    # The "weight" is implicitly determined by the optimizer via the centers and width, 
    # but this is a constrained model. 
    # Actually, if we just sum them, the amplitude is fixed at 1 for each peak.
    # This might be too restrictive.
    
    # Alternative: Use a linear combination of the Gaussians? No, we ran out of params.
    # Let's assume the sum of three unit-amplitude Gaussians is a reasonable starting point.
    # If the data requires different amplitudes, this model will be limited.
    
    y = g1 + g2 + g3
    
    return y
