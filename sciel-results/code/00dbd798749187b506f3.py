import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [0, 1] based on training ranges
    # b: [1, 5] -> (b-1)/4
    # s: [1, 5] -> (s-1)/4
    # temp: [1, 40] -> (temp-1)/39
    # pH: [2, 12] -> (pH-2)/10
    
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    t_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    p = params
    
    # Structural Change: Sparse Sum of Gaussian Basis Functions
    # Model: y = offset + sum_{i=1}^{4} A_i * exp(-((x - mu_i)^2) / (2 * sigma^2))
    # x is a 3D vector [t_norm, ph_norm, s_norm]
    # This allows capturing localized, multi-modal growth optima without rigid multiplicative constraints.
    
    # Parameters:
    # p[0]: Global Offset
    # p[1]: Global Width (sigma) - squared for positivity
    # p[2], p[3], p[4]: Center 1 (t, ph, s)
    # p[5], p[6], p[7]: Center 2 (t, ph, s)
    # p[8], p[9], p[10]... wait, only 10 params allowed.
    
    # Re-parameterization to fit 10 params:
    # We need 4 Gaussians. Each needs amplitude, center_t, center_ph, center_s.
    # That's 4*4 = 16 params. Too many.
    
    # Alternative: 2 Gaussians?
    # 2 * 4 = 8 params + offset + width = 10 params.
    # Let's try 2 Gaussian basis functions with shared isotropic width.
    # This is a significant structural change from the single Gaussian/Monod mix.
    # It allows for two distinct optimal regions or a more complex shape.
    
    # Parameters:
    # p[0]: Offset
    # p[1]: Shared Width (sigma) - squared for positivity
    # p[2]: Amplitude 1
    # p[3]: Center 1 t
    # p[4]: Center 1 ph
    # p[5]: Center 1 s
    # p[6]: Amplitude 2
    # p[7]: Center 2 t
    # p[8]: Center 2 ph
    # p[9]: Center 2 s
    
    sigma_sq = p[1]**2 + 1e-6
    
    # Gaussian 1
    A1 = p[2]
    c1_t = p[3]
    c1_ph = p[4]
    c1_s = p[5]
    
    # Gaussian 2
    A2 = p[6]
    c2_t = p[7]
    c2_ph = p[8]
    c2_s = p[9]
    
    # Compute distances squared in 3D space (t, ph, s)
    # d1^2 = (t_norm - c1_t)^2 + (ph_norm - c1_ph)^2 + (s_norm - c1_s)^2
    d1_sq = (t_norm - c1_t)**2 + (ph_norm - c1_ph)**2 + (s_norm - c1_s)**2
    g1 = np.exp(-d1_sq / (2 * sigma_sq))
    
    # d2^2 = (t_norm - c2_t)^2 + (ph_norm - c2_ph)^2 + (s_norm - c2_s)^2
    d2_sq = (t_norm - c2_t)**2 + (ph_norm - c2_ph)**2 + (s_norm - c2_s)**2
    g2 = np.exp(-d2_sq / (2 * sigma_sq))
    
    # Combine
    y = p[0] + A1 * g1 + A2 * g2
    
    # Ensure finite output
    if not np.all(np.isfinite(y)):
        return np.zeros_like(y)
        
    return y
