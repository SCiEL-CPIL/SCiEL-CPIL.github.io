"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a non-linear parametric model where the coefficients of a strain
    polynomial depend non-linearly on temperature.
    
    Model:
    sigma = A(T) + B(T)u + C(T)u^2 + D(T)u^3 + E(T)u^4
    
    Where:
    A(T) = p0 * exp(p1 * v)
    B(T) = p2 + p3 * v
    C(T) = p4 * exp(p5 * v)
    D(T) = p6 + p7 * v
    E(T) = p8 + p9 * v
    
    u = normalized strain
    v = normalized temperature
    
    This structure allows the curvature and offset of the stress-strain curve
    to vary exponentially with temperature, capturing softening/hardening
    effects more flexibly than a linear basis.
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
    
    # Normalize to [-1, 1]
    u = 2.0 * (strain - e_min) / e_range - 1.0
    v = 2.0 * (temp - t_min) / t_range - 1.0
    
    # Clip to [-1, 1] for stability
    u = np.clip(u, -1.0, 1.0)
    v = np.clip(v, -1.0, 1.0)
    
    p = params
    
    # Pre-compute powers of u
    u2 = u * u
    u3 = u2 * u
    u4 = u2 * u2
    
    # Calculate temperature-dependent coefficients
    # Using exp with clipping to prevent overflow/underflow issues during optimization
    # np.clip on the argument of exp helps stability if p1/p5 become large
    arg1 = np.clip(p[1] * v, -50, 50)
    arg5 = np.clip(p[5] * v, -50, 50)
    
    A_T = p[0] * np.exp(arg1)
    B_T = p[2] + p[3] * v
    C_T = p[4] * np.exp(arg5)
    D_T = p[6] + p[7] * v
    E_T = p[8] + p[9] * v
    
    # Construct stress
    stress = A_T + B_T * u + C_T * u2 + D_T * u3 + E_T * u4
    
    # Ensure finite output
    stress = np.where(np.isfinite(stress), stress, 0.0)
    
    return stress
# EVOLVE-BLOCK-END
