"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a separable Chebyshev polynomial structure:
    stress = sum_{d=0}^{4} (a_d + b_d * T_norm) * T_d(S_norm)
    
    where T_d is the d-th Chebyshev polynomial of the first kind,
    S_norm is strain normalized to [-1, 1], and T_norm is temperature
    normalized to [-1, 1].
    
    This is linear in the 10 parameters [a_0, b_0, a_1, b_1, ..., a_4, b_4].
    """
    
    # Normalization constants from training data ranges
    e_min = -0.001911
    e_max = 0.9636
    t_min = 0.06667
    t_max = 1.0
    
    e_range = e_max - e_min
    t_range = t_max - t_min
    
    # Ensure ranges are non-zero
    if e_range < 1e-12:
        e_range = 1.0
    if t_range < 1e-12:
        t_range = 1.0
    
    # Normalize to [-1, 1] for Chebyshev evaluation
    u = 2.0 * (strain - e_min) / e_range - 1.0
    v = 2.0 * (temp - t_min) / t_range - 1.0
    
    # Clip to [-1, 1] for stability (Chebyshev polynomials grow rapidly outside)
    u = np.clip(u, -1.0, 1.0)
    v = np.clip(v, -1.0, 1.0)
    
    # Pre-compute Chebyshev polynomials T_0 through T_4 for strain
    u2 = u * u
    u3 = u2 * u
    u4 = u2 * u2
    
    T0 = np.ones_like(u)
    T1 = u
    T2 = 2.0 * u2 - 1.0
    T3 = 4.0 * u3 - 3.0 * u
    T4 = 8.0 * u4 - 8.0 * u2 + 1.0
    
    # Temperature basis: linear in v (T_0 and T_1 for temp)
    v2 = v * v
    Tv0 = np.ones_like(v)
    Tv1 = v
    
    # Parameters: a_0, b_0, a_1, b_1, a_2, b_2, a_3, b_3, a_4, b_4
    p = params
    
    # Compute separable terms: (a_d + b_d * v) * T_d(u)
    term0 = (p[0] + p[1] * v) * T0
    term1 = (p[2] + p[3] * v) * T1
    term2 = (p[4] + p[5] * v) * T2
    term3 = (p[6] + p[7] * v) * T3
    term4 = (p[8] + p[9] * v) * T4
    
    stress = term0 + term1 + term2 + term3 + term4
    
    # Ensure finite output
    stress = np.where(np.isfinite(stress), stress, 0.0)
    
    return stress
# EVOLVE-BLOCK-END
