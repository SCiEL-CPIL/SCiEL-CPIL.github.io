import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod.
    Uses a hyperbolic tangent saturation model to capture elastic-plastic transition.
    This approach uses smooth, bounded non-linear terms instead of polynomials.
    """
    eps = strain
    T = temp
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Base terms: Linear + two Tanh saturation terms
    # Tanh terms capture the yield and hardening behavior
    term_lin = eps
    term_tanh1 = np.tanh(p3 * eps)
    term_tanh2 = np.tanh(p5 * eps)
    
    stress_base = p0 + p1 * term_lin + p2 * term_tanh1 + p4 * term_tanh2
    
    # Temperature dependent terms: Linear + Tanh interactions
    # The temperature effect is modeled as a linear function of T
    # applied to the same feature basis
    stress_temp = p6 + p7 * term_lin + p8 * term_tanh1 + p9 * term_tanh2
    
    return stress_base + T * stress_temp
