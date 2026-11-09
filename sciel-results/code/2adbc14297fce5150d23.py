"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Global rational function model using Chebyshev polynomials.
    
    f(x) = N(t) / D(t) where t = x/2
    N(t) = sum_{i=0}^{4} a_i T_i(t)
    D(t) = sum_{j=0}^{4} b_j T_j(t)
    
    This model can capture poles, asymptotes, and complex non-linear
    behaviors that linear basis expansions cannot easily represent.
    """
    # Scale input from [-2, 2] to [-1, 1] for Chebyshev evaluation
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 to T_4
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    # T_4(t) = 8t^4 - 8t^2 + 1
    
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t**2 - 1.0
    T3 = 4.0 * t**3 - 3.0 * t
    T4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
    
    # Extract parameters
    # First 5 for numerator, last 5 for denominator
    a = params[:5]
    b = params[5:]
    
    # Construct numerator and denominator
    # Using explicit linear combination for speed and clarity
    numerator = (a[0] * T0 + 
                 a[1] * T1 + 
                 a[2] * T2 + 
                 a[3] * T3 + 
                 a[4] * T4)
                 
    denominator = (b[0] * T0 + 
                   b[1] * T1 + 
                   b[2] * T2 + 
                   b[3] * T3 + 
                   b[4] * T4)
                   
    # Safety guard against division by zero
    # We add a tiny epsilon to the denominator to prevent inf/nan
    # This is a minimal perturbation that helps numerical stability
    # without significantly affecting the function shape for normal values
    epsilon = 1e-9
    # Using a smooth guard: if denominator is very small, nudge it away from 0
    # Simple addition can bias, so we use a conditional replacement for exact zeros
    # or very small values.
    # np.where is efficient enough for 200 points.
    # If |denom| < epsilon, replace with epsilon * sign(denom) (or epsilon if sign is 0)
    # To avoid discontinuity in gradients which might confuse BFGS, 
    # a simple addition of epsilon is often used in practice for robustness,
    # or we can just let it be if we trust the optimizer.
    # Given the start is all ones, denom is ~1 to 5. 
    # I will use a robust clip to prevent extreme values if the optimizer diverges.
    
    # Prevent division by zero
    # If denominator is 0, set to 1e-9
    # If denominator is very small, keep it but ensure no inf
    # A common robust approach:
    denom_safe = np.where(np.abs(denominator) < 1e-9, 
                          1e-9 * np.sign(denominator), 
                          denominator)
    # Handle the case where denominator is exactly 0 (sign is 0)
    denom_safe = np.where(denom_safe == 0, 1e-9, denom_safe)
    
    result = numerator / denom_safe
    
    return result
# EVOLVE-BLOCK-END
