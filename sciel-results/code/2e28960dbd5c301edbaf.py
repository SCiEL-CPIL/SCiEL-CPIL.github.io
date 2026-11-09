"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a normalized polynomial model.
    
    Normalizes inputs to improve optimization stability, then applies a 
    second-order polynomial with key interactions.
    """
    # Normalize inputs based on observed ranges to aid BFGS convergence
    # b: [1, 5] -> mean 3, std 1.5
    # s: [1, 5] -> mean 3, std 1.5
    # temp: [1, 40] -> mean 20, std 10
    # pH: [2, 12] -> mean 7, std 2
    
    b_n = (b - 3.0) / 1.5
    s_n = (s - 3.0) / 1.5
    t_n = (temp - 20.0) / 10.0
    ph_n = (pH - 7.0) / 2.0

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Linear terms
    pred = p0 * b_n + p1 * s_n + p2 * t_n + p3 * ph_n + p4
    
    # Quadratic terms for temperature and pH (curvature)
    pred += p5 * (t_n ** 2)
    pred += p6 * (ph_n ** 2)
    
    # Interaction terms
    pred += p7 * (b_n * t_n)
    pred += p8 * (s_n * t_n)
    pred += p9 * (b_n * ph_n)
    
    return pred
# EVOLVE-BLOCK-END
