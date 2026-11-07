import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Combines a power-law hardening core (Ramberg-Osgood style) with
    exponential saturation terms to better capture the elastic-plastic
    transition and post-yield plateau behavior of aluminium.
    
    Model:
        sigma = E(T) * eps + K(T) * |eps|^n * sign(eps) 
                + Y(T) * (1 - exp(-k(T) * |eps|)) * sign(eps)
                + polynomial corrections
    
    where:
        E(T) = p0 + p1 * T
        K(T) = p2 + p3 * T
        n = softplus(p4)
        Y(T) = p5 + p6 * T
        k(T) = max(0.1, p7 + p8 * T)
    
    Additional terms:
        + p9 * strain * temp
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Softplus for positive exponent: n = log(1 + exp(p4))
    p4_clamped = np.clip(p4, -50, 50)
    n = np.log1p(np.exp(p4_clamped))
    
    # Temperature-dependent coefficients
    E = p0 + p1 * temp          # Elastic modulus
    K = p2 + p3 * temp          # Hardening coefficient
    Y = p5 + p6 * temp          # Yield plateau stress
    k = p7 + p8 * temp          # Saturation sharpness
    
    # Ensure k is positive for numerical stability
    k_safe = np.maximum(k, 0.1)
    
    # Power-law hardening term: K * |eps|^n * sign(eps)
    abs_strain = np.abs(strain)
    sign_strain = np.sign(strain)
    
    # Compute |eps|^n safely
    power_term = np.power(abs_strain + 1e-15, n) * sign_strain
    
    # Exponential saturation term: Y * (1 - exp(-k * |eps|)) * sign(eps)
    # For small eps: ~ Y * k * eps (linear, adds to elastic slope)
    # For large eps: ~ Y * sign(eps) (plateau)
    exp_term = Y * (1.0 - np.exp(-k_safe * abs_strain)) * sign_strain
    
    # Main stress components
    elastic = E * strain
    hardening = K * power_term
    
    # Thermal-strain coupling
    thermal_coupling = p9 * strain * temp
    
    stress = elastic + hardening + exp_term + thermal_coupling
    
    return stress
