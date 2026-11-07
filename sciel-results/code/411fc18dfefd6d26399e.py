# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational function model: y = (a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4) / (1 + b1*x + b2*x^2 + b3*x^3 + b4*x^4)
    
    This uses 9 of the 10 parameters for the rational function coefficients.
    The 10th parameter is used as a global additive offset to help BFGS converge.
    
    Rational functions can capture poles, asymptotes, and hyperbolic behavior
    that linear basis expansions (Chebyshev, Fourier, RBF) cannot represent well.
    
    The denominator is constrained to be bounded away from zero on [0.1, 6]
    by adding a large constant to prevent numerical issues during optimization.
    """
    x = np.asarray(x, dtype=float).flatten()
    params = np.asarray(params, dtype=float).flatten()
    
    # Ensure exactly 10 parameters
    if len(params) < 10:
        params = np.concatenate([params, np.ones(10 - len(params))])
    else:
        params = params[:10]
    
    # Extract parameters
    # p0-p4: numerator coefficients (a0 to a4)
    # p5-p8: denominator coefficients (b1 to b4)
    # p9: global offset
    
    a0, a1, a2, a3, a4 = params[0], params[1], params[2], params[3], params[4]
    b1, b2, b3, b4 = params[5], params[6], params[7], params[8]
    offset = params[9]
    
    # Compute numerator
    num = a0 + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4
    
    # Compute denominator with a safety constant to avoid zero/near-zero denominators
    # The constant 10.0 helps keep the denominator bounded away from zero on [0.1, 6]
    # during optimization, while still allowing the rational structure to be learned.
    denom = 10.0 + b1 * x + b2 * x**2 + b3 * x**3 + b4 * x**4
    
    # Clip denominator to avoid division by zero or near-zero
    denom = np.where(np.abs(denom) < 1e-6, np.sign(denom) * 1e-6 + 1e-6, denom)
    
    y = num / denom + offset
    
    # Ensure finite output
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y
# EVOLVE-BLOCK-END
