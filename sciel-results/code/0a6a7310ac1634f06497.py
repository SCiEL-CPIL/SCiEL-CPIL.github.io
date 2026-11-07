import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model with smooth plateau temperature/pH response.
    
    Structure:
    V_max * Monod(s) * SmoothPlateau(temp) * SmoothPlateau(pH) * Inhibition(b) + Bias
    
    SmoothPlateau(x, center, half_width, sharpness):
    sigmoid((x - (center - half_width)) / sharpness) * sigmoid(((center + half_width) - x) / sharpness)
    
    Parameters:
    p0: log(K_s)
    p1: T_center shift (add to 37)
    p2: log(T_half_width)
    p3: log(k_t) (sharpness)
    p4: pH_center shift (add to 7)
    p5: log(pH_half_width)
    p6: log(k_p) (sharpness)
    p7: log(K_b)
    p8: log(V_max)
    p9: bias
    """
    eps = 1e-6
    
    # Extract parameters
    K_s = np.exp(params[0]) + eps
    T_center = 37.0 + params[1]
    T_hw = np.exp(params[2]) + eps
    k_t = np.exp(params[3]) + eps
    
    pH_center = 7.0 + params[4]
    pH_hw = np.exp(params[5]) + eps
    k_p = np.exp(params[6]) + eps
    
    K_b = np.exp(params[7]) + eps
    V_max = np.exp(params[8]) + eps
    bias = params[9]
    
    # Monod
    monod = s / (K_s + s)
    
    # Smooth Plateau for Temperature
    # Avoid overflow in sigmoid by clipping inputs
    z_t1 = (temp - (T_center - T_hw)) / k_t
    z_t2 = ((T_center + T_hw) - temp) / k_t
    
    # Stable sigmoid: 1 / (1 + exp(-x))
    # Use np.where to avoid warnings, but np.exp handles inf well if we clip
    # Actually, for large positive x, sigmoid -> 1. For large negative, -> 0.
    # We can compute sigmoid safely.
    sigmoid_t1 = 1.0 / (1.0 + np.exp(-np.clip(z_t1, -50, 50)))
    sigmoid_t2 = 1.0 / (1.0 + np.exp(-np.clip(z_t2, -50, 50)))
    
    temp_factor = sigmoid_t1 * sigmoid_t2
    
    # Smooth Plateau for pH
    z_p1 = (pH - (pH_center - pH_hw)) / k_p
    z_p2 = ((pH_center + pH_hw) - pH) / k_p
    
    sigmoid_p1 = 1.0 / (1.0 + np.exp(-np.clip(z_p1, -50, 50)))
    sigmoid_p2 = 1.0 / (1.0 + np.exp(-np.clip(z_p2, -50, 50)))
    
    pH_factor = sigmoid_p1 * sigmoid_p2
    
    # Density Inhibition
    inhib_b = K_b / (K_b + b)
    
    # Core
    core = V_max * monod * temp_factor * pH_factor * inhib_b
    
    result = core + bias
    
    # Ensure finite
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
