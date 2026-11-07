"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model: Ramberg-Osgood approximation with temperature-dependent properties.
    
    stress = E(T) * strain + sigma_y(T) * (strain / eps_y(T))**n
    
    Where:
    E(T) = p0 + p1*T       (Elastic Modulus)
    sigma_y(T) = p2 + p3*T (Yield Stress)
    eps_y(T) = p4 + p5*T   (Yield Strain)
    n = p6                 (Hardening Exponent)
    p7, p8, p9 are auxiliary constants for potential refinement or offset.
    
    To ensure smoothness and avoid issues with negative bases in power laws,
    we use a smooth approximation for the plastic term.
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, n, p7, p8, p9 = params
    
    # Temperature-dependent properties
    # Ensure E is positive
    E = p0 + p1 * temp
    E = np.maximum(E, 1e-6) # Safety clamp for optimization stability
    
    # Yield stress and strain
    sigma_y = p2 + p3 * temp
    eps_y = p4 + p5 * temp
    eps_y = np.maximum(eps_y, 1e-6) # Safety clamp
    
    # Ramberg-Osgood: epsilon = sigma/E + (sigma/sigma_y)^n / E
    # Inverting for sigma is complex. Instead, we use a direct stress formulation
    # that mimics the shape:
    # sigma = E * eps + k * (eps - eps_y_offset)^n * sign(eps - eps_y_offset)
    # But for differentiability and BFGS, a smooth power law is better.
    
    # Let's use a form: sigma = E * strain + sigma_y * (strain / eps_y)^n
    # This is not strictly Ramberg-Osgood inversion but captures the hardening.
    # However, for small strains, (strain/eps_y)^n might be unstable if strain < 0 and n is non-integer.
    # Aluminium data includes negative strain (compression).
    # We must handle negative strain.
    
    # Symmetric hardening:
    # sigma = E * strain + sigma_y * sign(strain) * (|strain| / eps_y)^n
    
    abs_strain = np.abs(strain)
    sign_strain = np.sign(strain)
    
    # Avoid 0^0 or negative base issues
    base = np.maximum(abs_strain / eps_y, 1e-10)
    
    # Hardening term
    hardening = sigma_y * (base ** n)
    
    # Elastic term
    elastic = E * strain
    
    # Total stress
    stress = elastic + sign_strain * hardening
    
    # Add small offset if needed (p9)
    stress = stress + p9
    
    return stress
# EVOLVE-BLOCK-END
