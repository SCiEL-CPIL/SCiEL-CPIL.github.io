"""Rewritten symbolic regression: stress-strain of aluminium rod with temperature dependence."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model: stress = E(T) * strain + plastic_correction
    where E(T) = E0 * (1 - c1*(1-T) - c2*(1-T)^2) captures temperature softening
    and plastic_correction = A * strain^n captures non-linear plastic behaviour.
    
    params[0]: E0 (reference elastic modulus)
    params[1]: c1 (linear temp softening coefficient)
    params[2]: c2 (quadratic temp softening coefficient)
    params[3]: A (plastic amplitude)
    params[4]: n (plastic exponent)
    params[5]: offset (baseline stress offset)
    params[6]: yield_strain (strain at which plastic flow begins)
    params[7]: hardening_rate (strain hardening coefficient)
    params[8]: temp_plastic_coupling (temperature effect on plasticity)
    params[9]: quadratic_strain_term (additional strain^2 term)
    """
    s = strain
    t = temp
    p = params
    
    # Temperature-dependent elastic modulus
    # Normalize temp: (1 - t) represents deviation from reference
    dt = 1.0 - t
    E = p[0] * (1.0 + p[1] * dt + p[2] * dt**2)
    
    # Elastic stress
    elastic = E * s
    
    # Plastic correction: activates beyond yield strain
    # Use smooth transition to avoid non-differentiable points
    yield_s = p[6]
    # Smooth activation: tanh-based
    activation = np.tanh(p[7] * (s - yield_s))
    
    # Plastic stress component
    plastic_base = p[3] * np.power(np.abs(s), p[4]) * np.sign(s)
    # Temperature coupling on plasticity
    temp_plastic = p[8] * dt * plastic_base
    # Strain hardening
    hardening = p[8] * s * np.abs(s)
    
    # Quadratic strain term
    quad_strain = p[9] * s**2
    
    # Combine
    stress = elastic + p[3] * activation * np.power(np.abs(s), p[4]) * np.sign(s) \
             + temp_plastic + hardening + quad_strain + p[5]
    
    return stress
# EVOLVE-BLOCK-END
