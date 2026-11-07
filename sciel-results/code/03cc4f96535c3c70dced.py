"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a smooth elastic-plastic model with temperature dependence.
    """
    eps = np.asarray(strain, dtype=np.float64)
    T = np.asarray(temp, dtype=np.float64)
    
    # Elastic regime: linear response
    E = params[0]  # Young's modulus
    
    # Yield strain (temperature dependent)
    eps_y = params[1] * (1.0 - params[2] * T)
    
    # Plastic hardening: smooth transition using tanh
    # This creates a smooth yield point
    plastic_strain = np.maximum(eps - eps_y, 0.0)
    
    # Hardening modulus (decreases with temperature)
    H = params[3] * (1.0 - params[4] * T)
    
    # Smooth yield using sigmoid-like transition
    transition_width = params[5]
    yield_factor = 0.5 * (1.0 + np.tanh((eps - eps_y) / transition_width))
    
    # Stress = elastic + plastic contribution
    # Elastic part: E * min(eps, eps_y) approximated smoothly
    elastic_stress = E * eps * (1.0 - yield_factor)
    # Plastic part: yield stress + H * plastic_strain
    yield_stress = E * eps_y
    plastic_stress = yield_stress + H * plastic_strain
    
    # Blend elastic and plastic
    stress = elastic_stress * (1.0 - yield_factor) + plastic_stress * yield_factor
    
    # Temperature softening
    temp_softening = 1.0 - params[6] * T
    stress = stress * temp_softening
    
    # Additional terms for better fit
    stress += params[7] * eps * T
    stress += params[8] * eps**2
    stress += params[9]
    
    return stress
# EVOLVE-BLOCK-END
