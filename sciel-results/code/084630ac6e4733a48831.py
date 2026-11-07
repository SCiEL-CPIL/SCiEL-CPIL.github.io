"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a smooth elastic-plastic model:
    - Elastic: stress = E_eff * strain
    - Plastic: stress = yield_stress + hardening * (strain - yield_strain)^n
    - Smooth transition via tanh blending
    - Temperature reduces both modulus and yield strength
    
    params (10 constants):
    p0: reference elastic modulus E0
    p1: temp coefficient for modulus (reduces E with temp)
    p2: reference yield stress sigma_y0
    p3: temp coefficient for yield stress
    p4: hardening coefficient
    p5: hardening exponent n
    p6: transition width (smoothness)
    p7: additional hardening scale / saturation
    p8: stress offset
    p9: temperature reference offset
    """
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    eps = 1e-12
    
    # Temperature effects
    t = temp - p9
    E_eff = p0 * (1.0 - p1 * t)
    sigma_y = p2 * (1.0 - p3 * t)
    
    # Ensure positive modulus and yield stress for numerical stability
    E_eff = np.maximum(E_eff, eps)
    sigma_y = np.maximum(sigma_y, eps)
    
    # Yield strain
    eps_y = sigma_y / E_eff
    
    # Strain magnitude and sign
    s = np.abs(strain)
    sign = np.sign(strain)
    
    # Smooth transition: blend elastic and plastic responses
    # t_param controls how sharp the transition is
    t_width = np.maximum(p6, eps)
    
    # Transition variable: how far into plastic regime
    # x = (s - eps_y) / t_width
    x = (s - eps_y) / t_width
    
    # Smooth step function: 0 for x<<0, 1 for x>>0
    # Using tanh: 0.5*(1 + tanh(x))
    step = 0.5 * (1.0 + np.tanh(x))
    
    # Elastic stress
    stress_elastic = E_eff * strain
    
    # Plastic stress: yield + hardening
    # plastic_strain = max(0, s - eps_y)
    # For smoothness, use softplus-like: t_width * log(1 + exp(x))
    # But simpler: just use max with smooth approximation
    plastic_strain = np.maximum(s - eps_y, 0.0)
    
    # Hardening term: p4 * plastic_strain^p5, with saturation
    # To avoid issues with p5 not being integer, use log-based power
    hardening = p4 * np.power(plastic_strain + eps, p5)
    
    # Saturation: limit hardening to p7
    hardening = hardening / (1.0 + hardening / (p7 + eps))
    
    stress_plastic = sigma_y + hardening
    
    # Blend: stress = (1-step)*elastic + step*plastic
    # But we need to handle sign correctly
    stress_mag = (1.0 - step) * s * E_eff + step * stress_plastic
    
    stress = sign * stress_mag + p8
    
    return stress
# EVOLVE-BLOCK-END
