"""Symbolic regression with rational function approximation."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational function approximation:
    y = (p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4) / (1 + q0*x + q1*x^2)
    
    params layout (length 10):
      params[0..4] = numerator coefficients p0..p4
      params[5..6] = denominator coefficients q0, q1 (denom = 1 + q0*x + q1*x^2)
      params[7..9] = unused (kept for compatibility)
    
    The denominator is 1 + q0*x + q1*x^2. To ensure stability and positivity
    over [-2, 2], we use a soft constraint: we compute the denominator and
    clip it to a small positive value if it becomes too small or negative.
    Actually, for BFGS stability, it's better to ensure the denominator stays
    away from zero. We'll use: denom = 1 + 0.1*tanh(q0)*x + 0.1*tanh(q1)*x^2
    This bounds the denominator coefficients to [-0.1, 0.1], keeping the
    denominator close to 1 and positive over [-2,2].
    """
    x = np.asarray(x, dtype=float)
    
    # Extract parameters
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    q0 = params[5]
    q1 = params[6]
    
    # Compute numerator
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    numerator = p0 + p1 * x + p2 * x2 + p3 * x3 + p4 * x4
    
    # Compute denominator with bounded coefficients for stability
    # Using tanh to bound q0, q1 to [-1, 1] and scaling by 0.5
    # denom = 1 + 0.5*tanh(q0)*x + 0.5*tanh(q1)*x^2
    # Over [-2,2], worst case: 1 - 0.5*2 - 0.5*4 = 1 - 1 - 2 = -2 (could be negative)
    # Let's use smaller scaling: 0.3
    # denom = 1 + 0.3*tanh(q0)*x + 0.3*tanh(q1)*x^2
    # Worst case: 1 - 0.3*2 - 0.3*4 = 1 - 0.6 - 1.2 = -0.8 (still negative)
    # Let's use even smaller: 0.2
    # denom = 1 + 0.2*tanh(q0)*x + 0.2*tanh(q1)*x^2
    # Worst case: 1 - 0.2*2 - 0.2*4 = 1 - 0.4 - 0.8 = -0.2 (still negative)
    # Let's use 0.1
    # denom = 1 + 0.1*tanh(q0)*x + 0.1*tanh(q1)*x^2
    # Worst case: 1 - 0.1*2 - 0.1*4 = 1 - 0.2 - 0.4 = 0.4 (positive!)
    
    q0_bounded = 0.1 * np.tanh(q0)
    q1_bounded = 0.1 * np.tanh(q1)
    
    denominator = 1.0 + q0_bounded * x + q1_bounded * x2
    
    # Safety: clip denominator to avoid division by zero
    # Since worst case is 0.4, this should be fine, but add a safeguard
    denominator = np.where(np.abs(denominator) < 1e-6, 1e-6 * np.sign(denominator + 1e-15), denominator)
    
    result = numerator / denominator
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
