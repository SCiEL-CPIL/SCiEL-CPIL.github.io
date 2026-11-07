"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model:
    stress = (E0 + E1*T + E2*T^2) * (a1*eps + a2*eps^2 + a3*eps^3 + a4*eps^4 + a5*eps^5) + L*eps + C
    
    Pure polynomial in strain (no division) for better numerical conditioning.
    
    Params:
    0: E0
    1: E1
    2: E2
    3: a1
    4: a2
    5: a3
    6: a4
    7: a5
    8: L
    9: C
    """
    # Extract parameters
    E0, E1, E2 = params[0], params[1], params[2]
    a1, a2, a3, a4, a5 = params[3], params[4], params[5], params[6], params[7]
    L, C = params[8], params[9]
    
    eps = strain
    T = temp
    
    # Temperature-dependent modulus
    E_T = E0 + E1 * T + E2 * T * T
    
    # Polynomial in strain (pure polynomial, no division)
    eps2 = eps * eps
    eps3 = eps2 * eps
    eps4 = eps2 * eps2
    eps5 = eps4 * eps
    strain_poly = a1 * eps + a2 * eps2 + a3 * eps3 + a4 * eps4 + a5 * eps5
    
    # Combine
    result = E_T * strain_poly + L * eps + C
    
    return result
# EVOLVE-BLOCK-END
