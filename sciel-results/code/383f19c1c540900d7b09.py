# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model: Ramberg-Osgood style with temperature dependence
    stress = E(T) * strain + H(T) * strain^n / (1 + k*strain) + C
    """
    E0, E1 = params[0], params[1]
    H0, H1 = params[2], params[3]
    n, k = params[4], params[5]
    C0, C1 = params[6], params[7]
    yield_T, yield_E = params[8], params[9]
    
    # Temperature-dependent Young's modulus
    E = E0 + E1 * temp
    
    # Temperature-dependent hardening
    H = H0 + H1 * temp
    
    # Nonlinear strain term: captures yield and hardening
    # strain^n / (1 + k*strain) gives sub-linear growth after initial linear region
    # For small strain, this ~ strain^n; for large strain, ~ strain^(n-1)/k
    nonlinear = np.power(np.abs(strain), n) * np.sign(strain) / (1.0 + k * np.abs(strain))
    
    # Yield transition: smooth step that reduces stiffness near yield
    # This creates a plateau-like behavior
    yield_factor = 1.0 / (1.0 + np.exp(yield_E * (np.abs(strain) - yield_T)))
    
    # Combine: elastic part modulated by yield + hardening + offset
    stress = E * strain * yield_factor + H * nonlinear + C0 + C1 * temp
    
    return stress
# EVOLVE-BLOCK-END
