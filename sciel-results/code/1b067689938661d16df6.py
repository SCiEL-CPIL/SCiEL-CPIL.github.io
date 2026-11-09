"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration of a damped nonlinear oscillator with driving force, from position and velocity."""
    # params: p0..p9 (length 10)
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Nonlinear damping term: base damping modulated by velocity
    # damp_eff = p0 + p1 * tanh(p2 * v)
    damp_eff = p0 + p1 * np.tanh(p2 * v)
    
    # Nonlinear stiffness term: base stiffness modulated by x^2 and v
    # stiff_eff = p3 + p4 * tanh(p5 * x**2 + p6 * v)
    stiff_eff = p3 + p4 * np.tanh(p5 * x**2 + p6 * v)
    
    # Driving / offset terms: constant + linear in x and v
    drive = p7 + p8 * x + p9 * v

    # Acceleration = -stiff_eff * x - damp_eff * v + drive
    acc = -stiff_eff * x - damp_eff * v + drive

    return acc
# EVOLVE-BLOCK-END
