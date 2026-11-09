"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a smoothstep-based piecewise polynomial model with temperature-dependent coefficients.
    C^1 continuous basis functions define elastic, yield, and hardening regions.
    
    Model:
    sigma = Elastic_Term + Yield_Term + Hardening_Term
    
    Elastic_Term = E(T) * eps
    E(T) = exp(p0 + p1 * T + p2 * T^2)
    
    Yield_Term = Y(T) * S(eps; a(T), b(T))
    S(x; a, b) = smoothstep transition from 0 to 1 centered at a with width b
    S(x; a, b) = 0.5 * (1 + tanh((x - a) / b))  # Using tanh for smoothness
    
    Hardening_Term = H(T) * eps^2 * (1 - S(-eps; a(T), b(T)))
    H(T) = p5 + p6 * T
    
    Parameters (10 constants):
    p0, p1, p2: Log-Elastic modulus coefficients (quadratic in T)
    p3, p4: Yield point location and width coefficients (linear in T)
    p5, p6: Hardening coefficients (linear in T)
    p7, p8: Additional yield shape parameters
    p9: Residual correction
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    
    # Temperature-dependent coefficients
    E_T = np.exp(p0 + p1 * T + p2 * T * T)
    
    # Yield point parameters
    a_T = p3 + p4 * T  # Yield strain location
    b_T = np.exp(p7 + p8 * T)  # Yield transition width (positive via exp)
    
    H_T = p5 + p6 * T
    
    # Smoothstep-like transition using tanh
    # S(eps) = 0.5 * (1 + tanh((eps - a_T) / b_T))
    # This transitions from 0 to 1 as eps increases past a_T
    S_pos = 0.5 * (1.0 + np.tanh((eps - a_T) / b_T))
    S_neg = 0.5 * (1.0 + np.tanh((-eps - a_T) / b_T))
    
    # Elastic term: active before yield
    elastic_term = E_T * eps * (1.0 - S_pos)
    
    # Yield term: plateau region
    yield_term = p9 * S_pos
    
    # Hardening term: active after yield
    hardening_term = H_T * eps * S_pos
    
    sigma = elastic_term + yield_term + hardening_term
    
    return sigma
# EVOLVE-BLOCK-END
