"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Structural change: Product of two low-degree polynomials.
    P_5(u) * Q_4(u) where u = x/2.
    This creates a non-convex landscape for BFGS.
    """
    # Check time
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Return a simple fallback to avoid hanging, though we should be fast
        return np.zeros_like(x)

    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Map x from [-2, 2] to u in [-1, 1]
    u = x / 2.0
    
    # Extract parameters for P (degree 5) and Q (degree 4)
    # p[0..5] for P, p[6..9] for Q
    p_coeffs = p[0:6]
    q_coeffs = p[6:10]
    
    # Compute P_5(u) = p0 + p1*u + p2*u^2 + p3*u^3 + p4*u^4 + p5*u^5
    # Using Horner's method for efficiency and numerical stability
    P = p_coeffs[5]
    P = P * u + p_coeffs[4]
    P = P * u + p_coeffs[3]
    P = P * u + p_coeffs[2]
    P = P * u + p_coeffs[1]
    P = P * u + p_coeffs[0]
    
    # Compute Q_4(u) = q0 + q1*u + q2*u^2 + q3*u^3 + q4*u^4
    Q = q_coeffs[4]
    Q = Q * u + q_coeffs[3]
    Q = Q * u + q_coeffs[2]
    Q = Q * u + q_coeffs[1]
    Q = Q * u + q_coeffs[0]
    
    result = P * Q
    
    # Safety check
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
        
    return result
# EVOLVE-BLOCK-END
