"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod.
    Uses a polynomial expansion in strain with temperature interactions.
    The model is linear in the parameters, ensuring robust convergence with BFGS.
    
    Model:
    sigma = p0 + p1*eps + p2*eps^2 + p3*eps^3 + p4*eps^4 + p5*eps^5
          + T * (p6 + p7*eps + p8*eps^2 + p9*eps^3)
    """
    eps = strain
    T = temp
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute powers of strain
    eps2 = eps * eps
    eps3 = eps2 * eps
    eps4 = eps2 * eps2
    eps5 = eps4 * eps
    
    # Base strain terms
    base = p0 + p1 * eps + p2 * eps2 + p3 * eps3 + p4 * eps4 + p5 * eps5
    
    # Temperature interaction terms
    temp_term = T * (p6 + p7 * eps + p8 * eps2 + p9 * eps3)
    
    return base + temp_term
# EVOLVE-BLOCK-END
