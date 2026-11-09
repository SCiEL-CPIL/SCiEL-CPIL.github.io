# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature.
    
    Structural change: multiplicative interaction model.
    Decouples linear elastic scaling, temperature-dependent stiffness,
    and non-linear strain saturation into separate smooth factors.
    
    Form: sigma = (p0 + p1*s) * (p2 + p3*t) * (1 + p4*s^p5) / (1 + p6*s^p7 + p8*t^p9) + p10
    where s and t are normalized strain and temperature.
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs to [0, 1] range for better conditioning
    s = (strain + 0.002) / 0.966
    t = (temp - 0.06667) / 0.93333
    
    # Clip exponents to avoid numerical issues
    p5_c = np.clip(p5, 0.1, 10.0)
    p7_c = np.clip(p7, 0.1, 10.0)
    p9_c = np.clip(p9, 0.1, 10.0)
    
    # Linear elastic scaling factor
    elastic = p0 + p1 * s
    
    # Temperature-dependent stiffness modulation
    temp_mod = p2 + p3 * t
    
    # Non-linear strain saturation numerator
    sat_num = 1.0 + p4 * np.power(np.maximum(s, 1e-10), p5_c)
    
    # Denominator with strain and temperature effects
    den = 1.0 + p6 * np.power(np.maximum(s, 1e-10), p7_c) + p8 * np.power(np.maximum(t, 1e-10), p9_c)
    
    # Safe division
    den_safe = np.where(np.abs(den) < 1e-10, 1e-10 * np.sign(den), den)
    den_safe = np.where(den_safe == 0, 1e-10, den_safe)
    
    stress = elastic * temp_mod * sat_num / den_safe
    
    # Ensure finite
    stress = np.where(np.isfinite(stress), stress, 0.0)
    
    return stress
# EVOLVE-BLOCK-END
