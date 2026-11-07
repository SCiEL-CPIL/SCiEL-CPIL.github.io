"""Combined program: Chebyshev polynomial basis + rational refinement with trigonometric denominator."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model combining:
    1. Chebyshev polynomial basis (total degree <= 3) for smooth polynomial trends
    2. Rational refinement with trigonometric denominator for non-polynomial behaviors
    
    Structure: (Chebyshev polynomial numerator) / (1 + small trigonometric correction)
    
    This allows the model to capture:
    - Polynomial trends via the Chebyshev basis (well-conditioned)
    - Saturation, periodic, or rational-like behaviors via the denominator
    - The denominator is constrained to be near 1, preserving stability
    """
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 5s margin from 150s limit
    
    # Ensure inputs are 1D arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64)
    
    if params.shape[0] < 10:
        return np.zeros_like(x)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Check deadline early
    if time.time() - start_time > deadline - 10.0:
        # Return a simple polynomial approximation as fallback
        return p0 + p1 * x + p2 * y
    
    # Chebyshev polynomials
    T0x = np.ones_like(x)
    T1x = x
    T2x = 2.0 * x**2 - 1.0
    T3x = 4.0 * x**3 - 3.0 * x
    
    T0y = np.ones_like(y)
    T1y = y
    T2y = 2.0 * y**2 - 1.0
    T3y = 4.0 * y**3 - 3.0 * y
    
    # Numerator: Chebyshev polynomial basis (total degree <= 3)
    # 10 basis functions:
    # b0 = T0(x)*T0(y) = 1
    # b1 = T1(x)*T0(y) = x
    # b2 = T0(x)*T1(y) = y
    # b3 = T2(x)*T0(y) = 2x^2-1
    # b4 = T0(x)*T2(y) = 2y^2-1
    # b5 = T1(x)*T1(y) = x*y
    # b6 = T3(x)*T0(y) = 4x^3-3x
    # b7 = T0(x)*T3(y) = 4y^3-3y
    # b8 = T1(x)*T2(y) = x*(2y^2-1)
    # b9 = T2(x)*T1(y) = (2x^2-1)*y
    
    # We use the first 7 parameters for the numerator (polynomial part)
    # and the last 3 parameters for the denominator correction (trigonometric)
    
    numerator = (p0 * (T0x * T0y) + 
                 p1 * (T1x * T0y) + 
                 p2 * (T0x * T1y) + 
                 p3 * (T2x * T0y) + 
                 p4 * (T0x * T2y) + 
                 p5 * (T1x * T1y) + 
                 p6 * (T3x * T0y))
    
    # Denominator: 1 + small trigonometric correction
    # Using p7, p8, p9 for sin(x), sin(y), cos(x)*cos(y) terms
    # Scale by 0.1 to keep denominator near 1 for stability
    denom_correction = 0.1 * (p7 * np.sin(x) + p8 * np.sin(y) + p9 * np.cos(x) * np.cos(y))
    denominator = 1.0 + denom_correction
    
    # Safety: prevent division by zero or near-zero
    eps = 1e-8
    abs_denom = np.abs(denominator)
    # Smooth regularization to keep denominator away from zero
    reg_term = eps * (denominator / (abs_denom + eps))
    denom_safe = denominator + reg_term
    
    # Check deadline before final computation
    if time.time() - start_time > deadline - 5.0:
        # Fallback: just return the numerator (polynomial part)
        return numerator
    
    result = numerator / denom_safe
    
    # Check for non-finite values
    if not np.all(np.isfinite(result)):
        # Return the polynomial part as fallback
        return numerator
    
    return result
# EVOLVE-BLOCK-END
