"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational function approximation:
    y = (p0 + p1*x + p2*x^2 + p3*x^3) / (1 + q1*x + q2*x^2)
    
    params: [p0, p1, p2, p3, q1, q2, ...]
    The denominator is 1 + q1*x + q2*x^2, which is positive for small q values
    and helps BFGS stability. We use only the first 6 params for the rational
    part; remaining params are unused (or could be used for additional terms
    if needed, but keeping it simple for stability).
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    q1 = params[4]
    q2 = params[5]
    
    # Compute numerator: p0 + p1*x + p2*x^2 + p3*x^3
    x2 = x * x
    x3 = x2 * x
    numerator = p0 + p1 * x + p2 * x2 + p3 * x3
    
    # Compute denominator: 1 + q1*x + q2*x^2
    # Add a small constant to ensure positivity and avoid division by zero
    denominator = 1.0 + q1 * x + q2 * x2
    
    # Ensure denominator is not zero (add small epsilon for numerical stability)
    # Use np.where to handle potential zeros
    eps = 1e-10
    safe_denominator = np.where(np.abs(denominator) < eps, eps * np.sign(denominator + 1e-15), denominator)
    
    result = numerator / safe_denominator
    
    # Ensure result is finite
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
