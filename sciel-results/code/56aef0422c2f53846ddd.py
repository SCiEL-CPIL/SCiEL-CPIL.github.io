"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model with sinusoidal drive.
    
    Model:
    a = p0 * sin(p1 * t + p2) + p3 * x + p4 * x^3 + p5 * v + p6 * v^3 + p7 * x^2 * v + p8 * x * v + p9
    
    Parameters:
    p0: Drive amplitude
    p1: Drive frequency
    p2: Drive phase
    p3: Linear stiffness
    p4: Nonlinear stiffness (Duffing term)
    p5: Linear damping
    p6: Nonlinear damping
    p7: Velocity-position coupling (Van der Pol term)
    p8: Velocity-position coupling
    p9: Constant bias
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Driving force
    drive = p0 * np.sin(p1 * t + p2)
    
    # Nonlinear restoring forces
    restoring = p3 * x + p4 * (x ** 3)
    
    # Damping forces
    damping = p5 * v + p6 * (v ** 3)
    
    # Coupling terms
    coupling = p7 * (x ** 2) * v + p8 * x * v
    
    # Bias
    bias = p9
    
    return drive + restoring + damping + coupling + bias
# EVOLVE-BLOCK-END
