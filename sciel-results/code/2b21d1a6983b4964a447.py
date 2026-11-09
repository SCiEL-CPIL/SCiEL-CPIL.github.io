# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # 5s safety margin before 150s kill

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature.
    
    Multiplicative interaction model with smooth yield transition:
    sigma = (p0 + p1*strain) * (p2 + p3*temp) * (1 + p4*strain^p5) / (1 + p6*strain^p7 + p8*temp^p9) + p10*tanh(p11*(strain - p12))
    
    Wait - only 10 params allowed. Let me restructure:
    sigma = (p0 + p1*strain) * (p2 + p3*temp) * (1 + p4*strain^p5) / (1 + p6*strain^p7 + p8*temp^p9) + p9
    
    Using raw inputs for better gradient flow, with safe power handling.
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Use raw inputs but ensure positive bases for power operations
    # Strain range: [-0.001911, 0.9636], Temp range: [0.06667, 1]
    # Shift strain to be positive for power operations
    s_base = strain + 0.01  # Now in [~0.008, ~0.974], all positive
    t_base = temp + 0.01    # Now in [~0.077, ~1.01], all positive
    
    # Clip to avoid extreme values
    s_base = np.clip(s_base, 1e-6, 10.0)
    t_base = np.clip(t_base, 1e-6, 10.0)
    
    # Ensure exponents are well-behaved
    # Use softplus-like transformation to keep exponents in reasonable range
    p5_safe = 0.5 * (np.abs(p5) + 0.1)  # Scale down large exponents
    p7_safe = 0.5 * (np.abs(p7) + 0.1)
    p8_safe = 0.5 * (np.abs(p8) + 0.1)
    
    # Compute power terms with numerical safeguards
    # For large exponents and values < 1, power can underflow; for values > 1, can overflow
    # Use log-space computation where possible
    log_s = np.log(s_base)
    log_t = np.log(t_base)
    
    # s^p5 = exp(p5 * log(s))
    exp_s_p5 = p5_safe * log_s
    exp_s_p7 = p7_safe * log_s
    exp_t_p8 = p8_safe * log_t
    
    # Clip exponent arguments to avoid overflow/underflow
    exp_s_p5 = np.clip(exp_s_p5, -50, 50)
    exp_s_p7 = np.clip(exp_s_p7, -50, 50)
    exp_t_p8 = np.clip(exp_t_p8, -50, 50)
    
    s_p5 = np.exp(exp_s_p5)
    s_p7 = np.exp(exp_s_p7)
    t_p8 = np.exp(exp_t_p8)
    
    # Multiplicative model components
    elastic = p0 + p1 * strain
    temp_factor = p2 + p3 * temp
    saturation_num = 1.0 + p4 * s_p5
    den = 1.0 + p6 * s_p7 + p8 * t_p8
    
    # Safe division
    den_safe = np.where(np.abs(den) < 1e-10, 1e-10 * np.sign(den), den)
    den_safe = np.where(den_safe == 0, 1e-10, den_safe)
    
    stress = elastic * temp_factor * saturation_num / den_safe + p9
    
    # Ensure finite
    stress = np.where(np.isfinite(stress), stress, 0.0)
    
    return stress
# EVOLVE-BLOCK-END
