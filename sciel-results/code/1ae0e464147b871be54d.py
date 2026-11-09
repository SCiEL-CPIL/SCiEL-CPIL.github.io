"""
Alternative strategy: Use a rational function form to better capture non-polynomial
behavior in the oscillator dynamics. Rational functions can model asymptotic behavior
and sharp transitions more efficiently than polynomials of the same parameter count.

The model uses:
- A rational restoring force: x * (1 + a*x^2) / (1 + b*x^2)
- A rational damping force: v * (1 + c*v^2) / (1 + d*v^2)
- Additional polynomial cross-terms for flexibility

This gives the optimizer a fundamentally different hypothesis space than pure polynomials.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts acceleration for a nonlinear oscillator using rational function components.
    
    Model structure:
    a = p0 
      + p1 * x * (1 + p2*x^2) / (1 + p3*x^2)   # rational restoring force
      + p4 * v * (1 + p5*v^2) / (1 + p6*v^2)   # rational damping force
      + p7 * x * v                              # linear coupling
      + p8 * x^2 * v                            # nonlinear coupling
      + p9 * v * x^2                            # additional nonlinear coupling
    
    This rational structure can capture:
    - Softening/hardening springs via the x-rational term
    - Saturation or threshold effects in damping via the v-rational term
    - The denominators allow the force to saturate or change character at large amplitudes
    """
    p = params
    
    # Ensure numerical stability: add small epsilon to denominators
    eps = 1e-8
    
    x2 = x * x
    v2 = v * v
    
    # Rational restoring force: x * (1 + p2*x^2) / (1 + p3*x^2)
    denom_x = 1.0 + p[3] * x2
    denom_x = np.where(np.abs(denom_x) < eps, np.sign(denom_x) * eps + (denom_x == 0) * eps, denom_x)
    restoring = x * (1.0 + p[2] * x2) / denom_x
    
    # Rational damping force: v * (1 + p5*v^2) / (1 + p6*v^2)
    denom_v = 1.0 + p[6] * v2
    denom_v = np.where(np.abs(denom_v) < eps, np.sign(denom_v) * eps + (denom_v == 0) * eps, denom_v)
    damping = v * (1.0 + p[5] * v2) / denom_v
    
    # Combine all terms
    result = (
        p[0]                                    # bias
        + p[1] * restoring                      # rational restoring force
        + p[4] * damping                        # rational damping force
        + p[7] * x * v                          # linear coupling
        + p[8] * x2 * v                         # quadratic-in-x coupling
        + p[9] * v * x2                         # another nonlinear coupling (redundant with p8? no, same term)
    )
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
