"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using centered variables and a rich polynomial basis.
    Centers t around 40 to improve numerical conditioning for BFGS optimizer.
    """
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = params
    
    # Center time to improve numerical conditioning
    t_c = t - 40.0
    
    # Core nonlinear terms for oscillator dynamics
    # Duffing-like potential: linear + cubic restoring force
    restoring = c0 * x + c1 * x**3
    # Damping terms: linear + quadratic
    damping = c2 * v + c3 * v**2
    # Cross coupling terms
    coupling = c4 * x * v
    # Time-dependent driving (sinusoidal with two harmonics)
    drive = c5 * np.sin(t_c) + c6 * np.cos(t_c)
    # Additional nonlinear interactions
    nonlinear = c7 * x**2 * v + c8 * x * v**2
    # Constant offset
    offset = c9
    
    result = restoring + damping + coupling + drive + nonlinear + offset
    return result
# EVOLVE-BLOCK-END
