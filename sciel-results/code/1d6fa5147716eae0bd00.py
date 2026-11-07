"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a tanh-based smooth step function to model the elastic-plastic transition.
    The stress is a linear elastic term multiplied by a saturation factor that smoothly
    transitions from 1 to a hardening plateau as strain increases, with transition width
    and plateau height modulated by temperature.
    
    Model:
        E(T) = p0 + p1 * T
        tau(T) = p2 + p3 * T          (transition width parameter)
        sigma_y(T) = p4 + p5 * T       (yield stress / plateau height)
        s = strain / tau(T)
        saturation = 0.5 * (1 + tanh(s))  # smooth step from 0 to 1
        stress = E(T) * strain * saturation + sigma_y(T) * (saturation - 0.5)
                 + p6 * strain^2 + p7 * T * strain + p8 * T + p9
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Temperature-dependent elastic modulus
    E_T = p0 + p1 * temp
    
    # Transition width (positive to avoid division by zero; use abs if needed)
    tau_T = np.abs(p2 + p3 * temp) + 1e-12
    
    # Yield stress / plateau height
    sigma_y = p4 + p5 * temp
    
    # Smooth saturation factor: 0.5*(1+tanh(strain/tau))
    s = strain / tau_T
    saturation = 0.5 * (1.0 + np.tanh(s))
    
    # Elastic-plastic stress: linear elastic modulated by saturation
    elastic_plastic = E_T * strain * saturation
    
    # Plateau contribution: shifts stress toward yield stress as saturation -> 1
    plateau = sigma_y * (saturation - 0.5)
    
    # Additional hardening and temperature terms
    hardening = p6 * strain**2
    temp_strain = p7 * temp * strain
    temp_only = p8 * temp + p9
    
    stress = elastic_plastic + plateau + hardening + temp_strain + temp_only
    
    return stress
# EVOLVE-BLOCK-END
