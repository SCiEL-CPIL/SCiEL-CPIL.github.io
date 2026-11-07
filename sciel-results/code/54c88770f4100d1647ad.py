"""Improved symbolic regression for aluminium stress-strain with temperature."""
import numpy as np
import time


def _softplus(x):
    """Numerically stable softplus."""
    return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -50, 50))))


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a smooth Ramberg-Osgood type model:
    sigma = E(T) * strain + H(T) * softplus(strain - eps_y(T))^p + offset(T)
    
    with exponential transient hardening and temperature-dependent coefficients.
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Clamp inputs to reasonable range
    strain_c = np.clip(strain, -0.002, 1.0)
    temp_c = np.clip(temp, 0.0, 1.0)
    
    # Temperature-dependent elastic modulus: E(T) = p0 + p1*T + p2*T^2
    E_T = params[0] + params[1] * temp_c + params[2] * temp_c ** 2
    
    # Temperature-dependent yield strain: eps_y(T) = p3 + p4*T
    eps_y = params[3] + params[4] * temp_c
    
    # Temperature-dependent hardening modulus: H(T) = p5 + p6*T
    H_T = params[5] + params[6] * temp_c
    
    # Power-law exponent (clamped for stability)
    p_exp = np.clip(params[7], 0.1, 5.0)
    
    # Transient hardening rate
    k_trans = np.clip(params[8], -50, 50)
    
    # Base offset
    offset = params[9]
    
    # Plastic strain beyond yield (smoothly)
    plastic_strain = np.maximum(strain_c - eps_y, 0.0)
    
    # Power-law hardening: H(T) * (plastic_strain)^p
    # Use softplus-like smoothing at the hinge for BFGS friendliness
    # smooth_hinge(x) = softplus(x * alpha) / alpha, but simpler: use max with small epsilon
    # Actually, let's use a smooth approximation: x^p for x>0, 0 otherwise
    # For BFGS, we want smoothness. Use: (softplus(x * beta) / beta)^p where beta is large
    beta = 50.0  # large enough to approximate max but smooth
    smooth_plastic = _softplus(plastic_strain * beta) / beta
    hardening_power = H_T * np.power(smooth_plastic, p_exp)
    
    # Exponential transient hardening (captures initial rapid hardening)
    # sigma_transient = A * (1 - exp(-k * plastic_strain))
    x_trans = k_trans * plastic_strain
    x_trans_clamped = np.clip(x_trans, -50, 50)
    transient = params[5] * 0.5 * (1.0 - np.exp(-x_trans_clamped))
    
    # Temperature-dependent offset (captures thermal expansion effects)
    T_offset = params[1] * 0.1 * temp_c + params[2] * 0.01 * temp_c ** 2
    
    # Elastic component
    elastic = E_T * strain_c
    
    # Total stress
    stress = elastic + hardening_power + transient + offset + T_offset
    
    return stress
