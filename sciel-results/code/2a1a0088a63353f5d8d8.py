"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model with polynomial nonlinearities.
    
    Model:
    a = p0 * sin(p1 * t + p2) + p3 * x + p4 * x**3 + p5 * v + p6 * v**2 + p7 * x * v + p8 * x**2 + p9
    
    Parameters:
    p0: Drive amplitude
    p1: Drive frequency
    p2: Drive phase
    p3: Linear position coefficient (stiffness)
    p4: Cubic position coefficient (nonlinear stiffness)
    p5: Linear velocity coefficient (damping)
    p6: Quadratic velocity coefficient
    p7: Velocity-position coupling
    p8: Quadratic position coefficient
    p9: Constant bias
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Driving force
    drive = p0 * np.sin(p1 * t + p2)
    
    # Polynomial terms in x and v
    # Note: We include up to 3rd order for x and 2nd for v to capture typical oscillator dynamics
    poly_x = p3 * x + p4 * (x ** 3) + p8 * (x ** 2)
    poly_v = p5 * v + p6 * (v ** 2)
    cross = p7 * x * v
    
    # Total acceleration
    return drive + poly_x + poly_v + cross + p9
# EVOLVE-BLOCK-END
