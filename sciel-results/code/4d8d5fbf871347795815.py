"""Symbolic regression: recover an unknown function with 10 parameters."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Return predictions for input x using 10 parameters.
    
    Uses a hybrid basis: Chebyshev polynomials (for well-conditioned polynomial
    approximation) combined with key non-polynomial features (sin, cos, log, 1/x)
    to capture diverse functional forms.
    
    Basis: [T_0, T_1, T_2, T_3, T_4, sin(x), cos(x), log(1+x), 1/x, exp(-x)]
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Clamp x to safe range (x is in [0.1, 6] per problem statement)
    x_safe = np.clip(x, 1e-6, 10.0)
    
    # Map x from [0.1, 6] to [-1, 1] for Chebyshev polynomials
    x_min = 0.1
    x_max = 6.0
    t = (2.0 * (x_safe - x_min) / (x_max - x_min)) - 1.0
    
    # Compute Chebyshev polynomials T_0 through T_4
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    
    # Non-polynomial basis functions
    sin_x = np.sin(x_safe)
    cos_x = np.cos(x_safe)
    log_x = np.log1p(x_safe)  # log(1+x), numerically stable
    inv_x = 1.0 / x_safe
    exp_neg_x = np.exp(-x_safe)
    
    # Linear combination: y = p0*T0 + p1*T1 + p2*T2 + p3*T3 + p4*T4
    #                     + p5*sin(x) + p6*cos(x) + p7*log(1+x)
    #                     + p8*(1/x) + p9*exp(-x)
    y = (p[0] * T0 +
         p[1] * T1 +
         p[2] * T2 +
         p[3] * T3 +
         p[4] * T4 +
         p[5] * sin_x +
         p[6] * cos_x +
         p[7] * log_x +
         p[8] * inv_x +
         p[9] * exp_neg_x)
    
    return y
# EVOLVE-BLOCK-END
