# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model: Ramberg-Osgood with smooth hinge transition
    sigma = E(T) * strain + H(T) * smooth_hinge(strain - eps_y(T))^p + offset + T^2 term
    
    params:
    [0]: E_base (Elastic modulus base)
    [1]: E_temp (Elastic modulus temp coeff)
    [2]: H_base (Hardening coeff base)
    [3]: H_temp (Hardening coeff temp coeff)
    [4]: eps_y_base (Yield strain base)
    [5]: eps_y_temp (Yield strain temp coeff)
    [6]: smoothness (delta for smooth hinge)
    [7]: power (hardening exponent)
    [8]: offset
    [9]: T2_coeff (Quadratic temp coeff)
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Unpack parameters
    E_base = params[0]
    E_temp = params[1]
    H_base = params[2]
    H_temp = params[3]
    eps_y_base = params[4]
    eps_y_temp = params[5]
    smoothness = params[6]
    power = params[7]
    offset = params[8]
    T2_coeff = params[9]
    
    # Clamp power to a reasonable range for hardening (typically 0.1 to 1.5)
    p = np.clip(power, 0.1, 1.5)
    
    # Clamp smoothness to avoid division by zero or huge values
    delta = np.clip(smoothness, 0.001, 0.1)
    
    # Temperature dependent coefficients
    E_T = E_base + E_temp * temp
    H_T = H_base + H_temp * temp
    eps_y_T = eps_y_base + eps_y_temp * temp
    
    # Calculate plastic strain component using smooth hinge
    x = strain - eps_y_T
    x_sq = x * x
    hinge = 0.5 * (x + np.sqrt(x_sq + delta * delta))
    
    # Ensure hinge is non-negative for power calculation if power is non-integer
    hinge_safe = np.maximum(hinge, 1e-12)
    hardening = H_T * (hinge_safe ** p)
    
    # Elastic component
    elastic = E_T * strain
    
    # Quadratic temp term
    t2_term = T2_coeff * temp * temp
    
    # Total stress
    stress = elastic + hardening + offset + t2_term
    
    return stress
# EVOLVE-BLOCK-END
