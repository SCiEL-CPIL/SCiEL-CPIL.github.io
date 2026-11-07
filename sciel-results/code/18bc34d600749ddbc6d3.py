"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a bilinear elastic-plastic model with:
    - Elastic region: stress = E * strain
    - Plastic region: stress = sigma_y + H * (strain - strain_y)
    - Temperature effects on modulus, yield stress, and hardening
    
    params (10 values):
    p0: Elastic modulus E
    p1: Yield stress sigma_y
    p2: Yield strain epsilon_y
    p3: Plastic hardening modulus H
    p4: Temperature coefficient for E (dE/dT)
    p5: Temperature coefficient for sigma_y (dsigma_y/dT)
    p6: Temperature coefficient for epsilon_y (deps/dT)
    p7: Temperature coefficient for H (dH/dT)
    p8: Smooth transition width (for tanh-based hinge)
    p9: Additional constant offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent parameters
    E = p0 + p4 * temp
    sigma_y = p1 + p5 * temp
    eps_y = p2 + p6 * temp
    H = p3 + p7 * temp
    
    # Ensure positive modulus and hardening
    E = np.maximum(E, 1e-6)
    H = np.maximum(H, 1e-6)
    
    # Elastic stress
    stress_elastic = E * strain
    
    # Plastic strain (using smooth hinge with tanh)
    # plastic_strain = max(0, strain - eps_y) approximated smoothly
    x = (strain - eps_y) / (p8 + 1e-10)
    plastic_strain = 0.5 * (x + np.tanh(x))
    plastic_strain = np.maximum(plastic_strain, 0.0)
    
    # Plastic stress contribution
    stress_plastic = sigma_y + H * plastic_strain
    
    # Combine: use elastic below yield, plastic above
    # Smooth blend using tanh
    blend_arg = (strain - eps_y) / (p8 + 1e-10)
    blend = 0.5 * (1 + np.tanh(blend_arg))
    
    stress = blend * stress_plastic + (1 - blend) * stress_elastic
    
    return stress + p9
# EVOLVE-BLOCK-END
