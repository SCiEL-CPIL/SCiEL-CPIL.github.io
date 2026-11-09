"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a hybrid model combining:
    1. A smooth saturating rational term (elastic-plastic knee)
    2. A piecewise-linear "sawtooth" basis (yield points / hardening segments)
    
    The rational term provides a smooth baseline that captures the initial
    elastic response and general saturation. The sawtooth basis adds kinked
    features to model discrete yield points and work-hardening regimes that
    are characteristic of aluminium stress-strain curves but difficult to
    capture with purely smooth functions.
    
    Model:
    sigma = Rational_Term + Sawtooth_Term
    
    Rational_Term = E(T) * eps / (1 + K(T) * |eps|)
    E(T) = exp(p0 + p1 * T)
    K(T) = exp(p2 + p3 * T)
    
    Sawtooth_Term = sum over i of A_i(T) * S_i(eps)
    where S_i(eps) = max(0, eps - y_i) - max(0, eps - y_{i+1})
    This creates a "tooth" that ramps up from y_i to y_{i+1} and then drops to 0.
    Alternatively, we can use a simpler kinked basis:
    S_i(eps) = max(0, eps - y_i)
    But a sawtooth is more flexible for oscillatory hardening.
    
    Let's use two sawtooth teeth:
    Tooth 1: from y1 to y2
    Tooth 2: from y3 to y4
    
    y_i are parameterized to be positive and in range.
    A_i(T) are amplitudes dependent on temperature.
    
    Parameters (10 constants):
    p0, p1: E(T) coefficients (log-linear in T)
    p2, p3: K(T) coefficients (log-linear in T)
    p4, p5: Amplitude of Tooth 1 (linear in T)
    p6, p7: Amplitude of Tooth 2 (linear in T)
    p8: Location of Tooth 1 start (scaled exp)
    p9: Location of Tooth 2 start (scaled exp)
    
    Note: We need 4 breakpoints (y1, y2, y3, y4) but only have 2 slots (p8, p9)
    if we keep 4 for E/K and 4 for Amplitudes.
    
    Let's restructure:
    p0, p1: E(T) = exp(p0 + p1*T)
    p2: K(T) = exp(p2)  (Constant K for simplicity, or linear?)
    Let's use K(T) = exp(p2 + p3*T) to keep flexibility.
    
    That leaves p4..p9 for Sawtooth.
    We can define the sawtooth teeth with fixed relative positions or parameterized.
    
    Let's try:
    Tooth 1: max(0, eps - y1) - max(0, eps - y2)
    Tooth 2: max(0, eps - y3) - max(0, eps - y4)
    
    We need y1, y2, y3, y4.
    And amplitudes A1, A2.
    Total 6 params.
    Plus E, K params (4). Total 10.
    
    p0, p1: E(T) = exp(p0 + p1*T)
    p2, p3: K(T) = exp(p2 + p3*T)
    p4: A1_base
    p5: A1_temp_coeff
    p6: A2_base
    p7: A2_temp_coeff
    p8: y1_scale (y1 = 0.01 * exp(p8))
    p9: y2_scale (y2 = 0.1 * exp(p9))
    
    Wait, we need 4 y's.
    Let's fix the "width" of the teeth or use a single kinked basis instead of sawtooth.
    
    Inspiration 2 used ReLU (kinks).
    Let's combine Rational (smooth) + ReLU (kinks).
    
    sigma = Rational + H1 * max(0, eps - y1) + H2 * max(0, eps - y2)
    
    Params:
    p0, p1: E(T)
    p2, p3: K(T)
    p4, p5: H1(T)
    p6, p7: H2(T)
    p8: y1
    p9: y2
    
    This fits 10 params perfectly.
    
    y1, y2 need to be positive.
    y1 = 0.005 * exp(p8)
    y2 = 0.05 * exp(p9)
    
    If p8=1, y1 ~ 0.0135
    If p9=1, y2 ~ 0.135
    
    This seems reasonable.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    
    # 1. Smooth Rational Term
    # E(T) and K(T) are positive
    E_T = np.exp(p0 + p1 * T)
    K_T = np.exp(p2 + p3 * T)
    
    abs_eps = np.abs(eps)
    den = 1.0 + K_T * abs_eps
    
    # Avoid division by zero if K_T is huge (though exp is bounded in practice for reasonable params)
    # If K_T is inf, den is inf, term is 0.
    rational_term = (E_T * eps) / den
    
    # 2. Kinked Piecewise-Linear Terms (ReLU basis)
    # Breakpoints
    # Scale factors chosen so that params=1 gives reasonable initial values
    y1 = 0.005 * np.exp(p8)
    y2 = 0.05 * np.exp(p9)
    
    # ReLU activations
    k1 = np.maximum(0.0, eps - y1)
    k2 = np.maximum(0.0, eps - y2)
    
    # Temperature-dependent slopes
    H1_T = p4 + p5 * T
    H2_T = p6 + p7 * T
    
    kink_term = H1_T * k1 + H2_T * k2
    
    sigma = rational_term + kink_term
    
    return sigma
# EVOLVE-BLOCK-END
