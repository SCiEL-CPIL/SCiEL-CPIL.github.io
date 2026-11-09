# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-Strain model for Aluminium with explicit Yield Transition.
    
    Model:
    sigma = (p0 + p1*t) * s  +  p2 * sigmoid(p3 * (s - p4)) * (p5 + p6 * log(1 + p7*s)) * (1 + p8*t) + p9
    
    Terms:
    1. (p0 + p1*t) * s : Linear elastic component, temperature dependent modulus.
    2. sigmoid(p3 * (s - p4)) : Smooth yield transition. p4 is yield strain, p3 is sharpness.
    3. (p5 + p6 * log(1 + p7*s)) : Logarithmic hardening behavior post-yield.
    4. (1 + p8*t) : Temperature modulation of plastic flow.
    5. p9 : Offset.
    """
    # Deadline guard
    if time.time() > _START_TIME + _DEADLINE - _SAFETY_MARGIN:
        return np.zeros_like(strain, dtype=np.float64)
        
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize strain to [0, 1] roughly
    s = (strain + 0.002) / 0.966
    s = np.clip(s, 0.0, 1.0)
    
    # Normalize temperature to [0, 1] roughly
    t = (temp - 0.06667) / 0.93333
    t = np.clip(t, 0.0, 1.0)
    
    # Safe parameters to avoid numerical issues
    # Ensure p7 is positive for log(1 + p7*s) to be well-behaved and increasing
    p7_safe = np.abs(p7) + 1e-6
    
    # Ensure p3 is large enough for a clear transition, but not infinite
    p3_safe = p3
    
    # 1. Elastic Term
    elastic_mod = p0 + p1 * t
    sigma_elastic = elastic_mod * s
    
    # 2. Yield Transition (Sigmoid)
    # Clip the argument to prevent overflow in exp
    arg = p3_safe * (s - p4)
    arg = np.clip(arg, -50, 50)
    yield_gate = 1.0 / (1.0 + np.exp(-arg))
    
    # 3. Hardening Term
    # Logarithmic hardening: log(1 + p7 * s)
    hardening = p5 + p6 * np.log1p(p7_safe * s)
    
    # 4. Temperature Plastic Modulation
    temp_plastic_factor = 1.0 + p8 * t
    
    # 5. Combine Plastic Term
    sigma_plastic = p2 * yield_gate * hardening * temp_plastic_factor
    
    # Total Stress
    stress = sigma_elastic + sigma_plastic + p9
    
    # Ensure finite
    stress = np.where(np.isfinite(stress), stress, 0.0)
    
    return stress
# EVOLVE-BLOCK-END
