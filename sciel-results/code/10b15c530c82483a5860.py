"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Damped driven Duffing oscillator model.
    
    Parameters (10 total):
    p0: linear damping coefficient
    p1: linear stiffness coefficient
    p2: cubic stiffness coefficient
    p3: quadratic stiffness coefficient
    p4: linear drive amplitude
    p5: drive frequency (omega)
    p6: drive phase (phi)
    p7: static offset
    p8: velocity damping nonlinear term coefficient (v^2)
    p9: cross-term coefficient (x*v)
    
    Model:
    a = -p0*v - p1*x - p2*x^3 - p3*x^2 + p4*sin(p5*t + p6) + p7 + p8*v^2 + p9*x*v
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Nonlinear oscillator terms
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    xv = x * v
    
    # Driving force
    drive = p4 * np.sin(p5 * t + p6)
    
    # Full acceleration
    a = (-p0 * v 
         - p1 * x 
         - p2 * x3 
         - p3 * x2 
         + drive 
         + p7 
         + p8 * v2 
         + p9 * xv)
    
    return a
# EVOLVE-BLOCK-END
