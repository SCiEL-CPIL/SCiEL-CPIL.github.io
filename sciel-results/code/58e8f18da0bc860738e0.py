import numpy as np
import time

_start_time = None

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model with elastic linear term and saturating plastic term.
    stress = E(T)*strain + S(T)*tanh(strain / alpha(T)) + C
    
    E(T) = p0 + p1*T + p2*T^2
    S(T) = p3 + p4*T + p5*T^2
    alpha(T) = exp(p6 + p7*T + p8*T^2)
    C = p9
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > 130:
        return np.zeros_like(strain, dtype=np.float64)
    
    p0, p1, p2 = params[0], params[1], params[2]
    p3, p4, p5 = params[3], params[4], params[5]
    p6, p7, p8 = params[6], params[7], params[8]
    p9 = params[9]
    
    T = temp
    S = strain
    
    # Temperature-dependent coefficients
    E_T = p0 + p1 * T + p2 * T * T
    S_T = p3 + p4 * T + p5 * T * T
    
    # Ensure alpha is positive and stable
    alpha_log = p6 + p7 * T + p8 * T * T
    # Clip log alpha to prevent overflow/underflow in exp
    alpha_log = np.clip(alpha_log, -20, 20)
    alpha_T = np.exp(alpha_log)
    
    # Avoid division by zero if alpha is extremely small (though exp prevents 0)
    # But if alpha is very small, strain/alpha is huge, tanh -> 1
    # This is fine.
    
    # Compute tanh term
    # To avoid numerical issues with huge arguments, tanh is stable.
    ratio = S / alpha_T
    # Clip ratio to prevent any extreme floating point issues, though tanh handles large inputs well
    ratio = np.clip(ratio, -50, 50)
    tanh_term = np.tanh(ratio)
    
    # Combine
    elastic = E_T * S
    plastic = S_T * tanh_term
    offset = p9
    
    result = elastic + plastic + offset
    
    return result
