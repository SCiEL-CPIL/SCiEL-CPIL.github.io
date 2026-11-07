import numpy as np
import time

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
    
    # Structural change: Sparse Sum of 4 Gaussian Basis Functions
    # Model: y = sum_{i=1}^{4} A_i * exp(- ||x - c_i||^2 / (2 * sigma^2))
    # Inputs for Gaussian: (t_norm, ph_norm, s_norm) 
    # Note: b_norm is excluded from the Gaussian centers to save parameters? 
    # The directive says "3D center (temperature, pH, substrate)". So b is not in the center.
    # But b must affect the output. 
    # Let's add a linear term for b_norm or include it in the amplitude?
    # Let's add a linear term for b_norm: y = base + p[0]*b_norm + sum(Gaussians)
    # Or maybe the amplitudes depend on b? 
    # Let's stick to the directive's structure first: 4 Gaussians with 3D centers (t, ph, s).
    # Parameters:
    # p[0]: Linear coefficient for b_norm (to account for biomass/batch effect linearly)
    # p[1]: Shared width sigma (inverse variance scale)
    # For each Gaussian i (0 to 3):
    #   p[2 + i*4] : Amplitude A_i
    #   p[3 + i*4] : Center t_i
    #   p[4 + i*4] : Center ph_i
    #   p[5 + i*4] : Center s_i
    # Total params: 1 + 1 + 4*4 = 18. 
    # Wait, limit is 10 params.
    # 4 Gaussians * (1 amp + 3 centers) = 16 params. Too many.
    # We need to reduce.
    # Option 1: 2 Gaussians. 2 * (1+3) = 8 params. Plus 2 for width and linear b. Total 10.
    # Option 2: 4 Gaussians with shared centers? No, that defeats the purpose.
    # Option 3: 4 Gaussians, but fix some centers or use fewer dimensions?
    # The directive says "sparse sum of four Gaussian basis functions". 
    # If I strictly follow "4 Gaussians", I cannot fit 4*4=16 params in 10 slots.
    # Unless... I share the width (1 param), and maybe share some centers? 
    # Or maybe the "3D center" implies 3 params per Gaussian, plus 1 amplitude = 4 params/Gaussian.
    # 4 Gaussians = 16 params. 
    # Constraint: "up to 10 fitted constants".
    # This is a contradiction in the directive vs constraint if interpreted literally as 4 independent Gaussians with full 3D centers and amplitudes.
    # 
    # Let's re-read: "sparse sum of four Gaussian basis functions, each parameterized by a 3D center... and a shared isotropic width".
    # Maybe the "sparse" part implies we can fix some? No, "fitted constants".
    # Perhaps I should use **2** Gaussians to fit in 10 params? 
    # 2 Gaussians: 
    #   Width: 1
    #   G1: Amp, Ct, Cph, Cs (4)
    #   G2: Amp, Ct, Cph, Cs (4)
    #   Total: 9.
    #   Plus 1 linear term for b_norm. Total 10.
    # This fits the constraint.
    # 
    # Alternatively, maybe the directive implies a *structure* that I should adapt to fit the param limit.
    # Let's try 2 Gaussians first. It's a structural change from the rational function.
    # 
    # Parameters:
    # p[0]: Linear coeff for b_norm
    # p[1]: Width scale (sigma_inv or sigma^2?) Let's use sigma (width). 
    #        To avoid division by zero, we can use softplus or clip. 
    #        Let's use p[1] as the variance inverse (precision) to keep it positive? No, BFGS doesn't enforce positivity.
    #        Let's use exp(p[1]) for the width to ensure it's positive.
    # p[2]: Amp 1
    # p[3]: Center t 1
    # p[4]: Center ph 1
    # p[5]: Center s 1
    # p[6]: Amp 2
    # p[7]: Center t 2
    # p[8]: Center ph 2
    # p[9]: Center s 2
    
    # Width
    sigma = np.exp(p[1])
    
    # Distances squared
    # x = (t_norm, ph_norm, s_norm)
    # c1 = (p[3], p[4], p[5])
    # c2 = (p[7], p[8], p[9])
    
    d1_t = t_norm - p[3]
    d1_ph = ph_norm - p[4]
    d1_s = s_norm - p[5]
    dist1_sq = d1_t**2 + d1_ph**2 + d1_s**2
    
    d2_t = t_norm - p[7]
    d2_ph = ph_norm - p[8]
    d2_s = s_norm - p[9]
    dist2_sq = d2_t**2 + d2_ph**2 + d2_s**2
    
    # Gaussians
    g1 = p[2] * np.exp(-0.5 * dist1_sq / (sigma**2))
    g2 = p[6] * np.exp(-0.5 * dist2_sq / (sigma**2))
    
    # Linear term for b
    linear_b = p[0] * b_norm
    
    y = linear_b + g1 + g2
    
    return y
