"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model: Driven Duffing with quadratic damping.
    
    a = p0*cos(p2*t) + p1*sin(p2*t) + p3*x + p4*x^3 + p5*v + p6*v^3 + p7*x*v + p8*v^2 + p9
    
    Parameters:
    p0: Cosine drive amplitude
    p1: Sine drive amplitude
    p2: Drive frequency
    p3: Linear stiffness
    p4: Cubic stiffness
    p5: Linear damping
    p6: Cubic damping
    p7: Velocity-position coupling
    p8: Quadratic damping
    p9: Constant bias
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Drive terms
    drive = p0 * np.cos(p2 * t) + p1 * np.sin(p2 * t)
    
    # Restoring force terms
    restoring = p3 * x + p4 * x**3
    
    # Damping terms
    damping = p5 * v + p6 * v**3 + p8 * v**2
    
    # Coupling and bias
    coupling = p7 * x * v
    bias = p9
    
    return drive + restoring + damping + coupling + bias
# EVOLVE-BLOCK-END
