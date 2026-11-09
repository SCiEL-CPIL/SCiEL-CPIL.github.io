"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration of a damped nonlinear oscillator with driving force, from position and velocity."""
    # params: p0..p9 (length 10)
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # State-dependent stiffness: base + quadratic in x, modulated by velocity
    # Duffing-like term: k_eff = p1 + p2 * x^2
    # Additional velocity-dependent modulation to capture complex nonlinearities
    stiffness_mod = 1.0 + p3 * np.tanh(p4 * v)
    k_eff = (p1 + p2 * x**2) * stiffness_mod

    # Damping: linear + nonlinear (cubic in v or quadratic in x)
    # c_eff = p0 + p5 * v**2 + p6 * x * v
    damping = p0 + p5 * v**2 + p6 * x * v

    # Driving / offset terms: constant + linear in x and v
    # Additional higher-order cross terms for flexibility
    drive = p7 + p8 * x + p9 * v

    # Acceleration = -k_eff * x - damping * v + drive
    acc = -k_eff * x - damping * v + drive

    return acc
# EVOLVE-BLOCK-END
