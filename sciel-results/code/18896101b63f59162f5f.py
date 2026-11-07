"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational function model for nonlinear oscillator acceleration.
    
    Structure:
    a = (Drive + Stiffness + Damping + Cross) / (1 + Quadratic_Denom)
    
    Parameters:
    p0, p1, p2: Drive amplitude, frequency, phase
    p3, p4: Linear and cubic stiffness coefficients
    p5, p6: Linear and cubic damping coefficients
    p7, p8: Quadratic denominator coefficients for x^2 and v^2
    p9: Cross-term coefficient for x*v
    """
    # Check time budget
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        # Return a safe fallback if we are running out of time
        # This shouldn't happen in normal evaluation unless the fit is extremely slow
        return np.zeros_like(x)

    p = params
    
    # Drive term: p0 * sin(p1 * t + p2)
    drive = p[0] * np.sin(p[1] * t + p[2])
    
    # Stiffness terms: p3 * x + p4 * x^3
    stiff = p[3] * x + p[4] * (x ** 3)
    
    # Damping terms: p5 * v + p[6] * v^3
    damp = p[5] * v + p[6] * (v ** 3)
    
    # Cross term: p9 * x * v
    cross = p[9] * x * v
    
    # Numerator
    numerator = drive + stiff + damp + cross
    
    # Denominator: 1 + p7 * x^2 + p8 * v^2
    # Add a small epsilon to prevent division by zero if parameters drive it there
    denominator = 1.0 + p[7] * (x ** 2) + p[8] * (v ** 2)
    
    # Safe division
    # If denominator is close to zero, it causes instability.
    # Clip denominator to avoid 0.
    # Since x, v are small, 1.0 dominates unless p7, p8 are large negative.
    # We use np.where to handle potential singularities gracefully
    safe_denom = np.where(np.abs(denominator) < 1e-6, 1e-6 * np.sign(denominator + 1e-12), denominator)
    
    return numerator / safe_denom
# EVOLVE-BLOCK-END
