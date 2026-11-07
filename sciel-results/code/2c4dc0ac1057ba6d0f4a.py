"""Rational function approximation for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational function: y = P(x) / Q(x)
    
    P(x) = p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4
    Q(x) = 1 + q0*x + q1*x^2 + q2*x^3 + q3*x^4
    
    The +1 in Q ensures the denominator is positive near x=0 when params start at 1,
    helping BFGS convergence. We also add a soft regularization to keep Q away from 0.
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
    q2 = params[7]
    q3 = params[8]
    # params[9] is unused but kept for interface compatibility
    
    # Compute powers
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    # Numerator
    num = p0 + p1 * x + p2 * x2 + p3 * x3 + p4 * x4
    
    # Denominator: 1 + q0*x + q1*x^2 + q2*x^3 + q3*x^4
    den = 1.0 + q0 * x + q1 * x2 + q2 * x3 + q3 * x4
    
    # Soft clip to avoid division by zero and keep values finite
    # Use tanh-based soft clipping to keep the ratio bounded
    # We want den to be bounded away from 0. Use a smooth floor.
    # den_safe = den + sign(den) * max(0, 0.1 - |den|)
    # Simpler: add a small constant to keep it away from 0
    # But that changes the function. Instead, use a softplus-like approach.
    
    # Better: use np.where to handle near-zero denominators smoothly
    # For BFGS, we need smoothness. Let's use:
    # effective_den = den + 0.01 * np.sign(den) where |den| < 0.1
    # But np.sign is not differentiable at 0.
    
    # Alternative: den_safe = den + 0.05 * (1 - np.tanh(den * 10)) * np.sign(den + 1e-8)
    # This is getting complex. Let's use a simpler approach:
    # Add a small constant that depends on the magnitude to push away from 0
    
    # Actually, the simplest robust approach: 
    # den_safe = np.where(np.abs(den) < 0.1, 0.1 * np.sign(den + 1e-10), den)
    # But this has a kink at |den|=0.1.
    
    # Let's use: den_safe = den + 0.01 * np.tanh(den * 50) * 0.0
    # That doesn't help.
    
    # Better idea: use the identity that for BFGS, as long as the function is 
    # smooth and finite, it will work. The main risk is den crossing 0.
    # With 1 + polynomial, and params starting at 1, den = 1 + x + x^2 + x^3 + x^4
    # which is always positive on [-2,2]. So BFGS should stay in a region where den > 0.
    # We just need to handle the case where BFGS might try values that make den close to 0.
    
    # Use a smooth floor: den_safe = den + 0.05 * np.maximum(0, 1 - np.abs(den)/0.05)
    # No, let's just use a simple additive term that vanishes when |den| is large:
    # den_safe = den + 0.01 * np.exp(-np.abs(den) * 10)
    # This is smooth and pushes den away from 0.
    
    den_safe = den + 0.01 * np.exp(-np.abs(den) * 10)
    
    result = num / den_safe
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
