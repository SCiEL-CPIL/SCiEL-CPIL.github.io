"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a sum-of-sines (Fourier) basis.
    
    Inputs:
        b: population density (range ~[1, 5])
        s: substrate concentration (range ~[1, 40])
        temp: temperature (range ~[1, 40])
        pH: pH level (range ~[1, 12])
    
    Params (10 values):
        p0: amplitude for temperature sine
        p1: frequency for temperature sine
        p2: phase for temperature sine
        p3: amplitude for pH sine
        p4: frequency for pH sine
        p5: phase for pH sine
        p6: linear coefficient for b (density)
        p7: linear coefficient for s (substrate)
        p8: bias term
        p9: amplitude for temperature cosine (secondary harmonic)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs to reasonable scales for numerical stability
    # Temperature: center around 30C (typical E. coli optimum), scale by 10
    t_norm = (temp - 30.0) / 10.0
    # pH: center around 7, scale by 2
    ph_norm = (pH - 7.0) / 2.0
    # Density: center around 3, scale by 2
    b_norm = (b - 3.0) / 2.0
    # Substrate: center around 10, scale by 10
    s_norm = (s - 10.0) / 10.0
    
    # Fourier basis for temperature (captures optimal temp window)
    # Use both sine and cosine with learnable frequency and phase
    t_arg = p1 * t_norm + p2
    t_basis = p0 * np.sin(t_arg) + p9 * np.cos(t_arg)
    
    # Fourier basis for pH (captures pH optimum)
    ph_arg = p4 * ph_norm + p5
    ph_basis = p3 * np.sin(ph_arg)
    
    # Linear terms for density and substrate
    # These capture monotonic or saturating effects
    linear_term = p6 * b_norm + p7 * s_norm
    
    # Combine all components
    result = t_basis + ph_basis + linear_term + p8
    
    return result
# EVOLVE-BLOCK-END
