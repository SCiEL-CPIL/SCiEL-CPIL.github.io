import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Sparse sum of four Gaussian basis functions in 3D space (temp, pH, s)
    # plus a biomass-dependent modulation and a bias term.
    # This allows the model to capture localized, multi-modal optima without
    # assuming a global mechanistic form.
    
    # Normalize inputs to [0, 1]
    # b: [1, 5] -> (b-1)/4
    # s: [1, 5] -> (s-1)/4
    # temp: [1, 40] -> (temp-1)/39
    # pH: [2, 12] -> (pH-2)/10
    
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    t_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    p = params
    
    # Parameters:
    # p[0]: bias
    # p[1]: shared Gaussian width (inverse std^2), mapped to positive
    # p[2], p[3], p[4]: center of basis 1 (t, ph, s)
    # p[5]: amplitude of basis 1
    # p[6], p[7], p[8]: center of basis 2 (t, ph, s)
    # p[9]: amplitude of basis 2
    # Wait, we only have 10 params. We need 4 bases.
    # 4 bases * 3 centers = 12 params. Too many.
    # We need a different structure or fewer bases.
    # Let's try 2 bases + biomass term + bias.
    # Or 1 basis + polynomial?
    # The directive says "sparse sum of four Gaussian basis functions".
    # But we only have 10 params.
    # 4 bases * 3 coords = 12. Plus 4 amplitudes = 16. Plus width = 17.
    # This is impossible with 10 params.
    
    # Let's re-read the directive: "sparse sum of four Gaussian basis functions, each parameterized by a 3D center... and a shared isotropic width".
    # Maybe the centers are fixed? No, "parameterized".
    # Maybe the inputs are not all 3D? "temperature, pH, substrate".
    # If we use 2 bases:
    # Base 1: c1t, c1ph, c1s, a1 (4 params)
    # Base 2: c2t, c2ph, c2s, a2 (4 params)
    # Width: w (1 param)
    # Bias: b0 (1 param)
    # Total: 10 params.
    # This fits perfectly. The directive says "four", but maybe it's a suggestion for a family, or maybe I can use a different encoding.
    # However, "Make a STRUCTURAL change... Replace ... with a sparse sum of four Gaussian basis functions".
    # If I can't fit 4, I should fit the closest structural change that is feasible.
    # Or maybe the biomass term is handled differently?
    # Let's stick to the feasible 2-basis Gaussian model + bias. This is still a structural change from the multiplicative mechanistic model.
    
    # Alternative: 1 basis in 4D? (b, s, t, ph).
    # Center: 4 params.
    # Amplitude: 1 param.
    # Width: 1 param.
    # Bias: 1 param.
    # Total: 7 params.
    # This is also a valid structural change (Gaussian process-like kernel).
    # Let's try the 2-basis 3D model (t, ph, s) as it separates biomass which often has a distinct inhibitory effect.
    
    # Structure:
    # y = b0 + a1 * exp(-w * ((t-c1t)^2 + (ph-c1ph)^2 + (s-c1s)^2)) * f(b) + a2 * exp(-w * ((t-c2t)^2 + (ph-c2ph)^2 + (s-c2s)^2)) * f(b)
    # What is f(b)? Maybe just 1.0? Or exp(-k*b)?
    # If f(b) is 1.0, we have 10 params exactly.
    # b0, a1, c1t, c1ph, c1s, a2, c2t, c2ph, c2s, w.
    
    # Let's try this.
    
    w = np.exp(p[9])  # Shared width, positive
    # Use softplus or exp for width to ensure positive
    
    # Basis 1
    c1t = p[2]
    c1ph = p[3]
    c1s = p[4]
    a1 = p[5]
    
    d1_sq = (t_norm - c1t)**2 + (ph_norm - c1ph)**2 + (s_norm - c1s)**2
    g1 = np.exp(-w * d1_sq)
    
    # Basis 2
    c2t = p[6]
    c2ph = p[7]
    c2s = p[8]
    a2 = p[1]  # Reusing indices? Let's map carefully.
    
    # Mapping:
    # p[0]: bias
    # p[1]: amp1
    # p[2]: c1t
    # p[3]: c1ph
    # p[4]: c1s
    # p[5]: amp2
    # p[6]: c2t
    # p[7]: c2ph
    # p[8]: c2s
    # p[9]: width (log)
    
    a1 = p[1]
    a2 = p[5]
    
    d2_sq = (t_norm - c2t)**2 + (ph_norm - c2ph)**2 + (s_norm - c2s)**2
    g2 = np.exp(-w * d2_sq)
    
    y = p[0] + a1 * g1 + a2 * g2
    
    return y
