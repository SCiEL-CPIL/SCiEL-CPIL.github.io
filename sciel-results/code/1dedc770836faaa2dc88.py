import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression for a nonlinear oscillator using a rational/sigmoidal basis.
    
    This strategy differs from the linear trigonometric basis by using a nonlinear
    denominator. The parameters p[0..9] directly scale the components of the numerator
    and denominator, creating a non-linear interaction between x and v that is not
    representable as a linear combination of fixed basis functions. This allows BFGS
    to optimize the shape of the function itself, not just its amplitude.
    
    Structure:
    N = p0 + p1*sin(x) + p2*cos(x) + p3*sin(v) + p4*cos(v) + p5*sin(x+v) + p6*cos(x+v) + p7*sin(x-v) + p8*cos(x-v) + p9*x*sin(v)
    D = 1 + p1*0.1*sin(x) + p2*0.1*cos(x) + p3*0.1*sin(v) + p4*0.1*cos(v) + p5*0.1*sin(x+v) + p6*0.1*cos(x+v) + p7*0.1*sin(x-v) + p8*0.1*cos(x-v) + p9*0.1*x*sin(v)
    
    Note: The small scaling (0.1) in the denominator ensures D is close to 1 initially
    and prevents division by zero, while still allowing significant non-linear modulation
    as BFGS adjusts the parameters.
    """
    p = params
    
    # Precompute trigonometric functions
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_v = np.sin(v)
    cos_v = np.cos(v)
    
    # Combined phase arguments
    sin_xv = np.sin(x + v)
    cos_xv = np.cos(x + v)
    sin_x_minus_v = np.sin(x - v)
    cos_x_minus_v = np.cos(x - v)
    
    # Interaction term
    x_sin_v = x * sin_v
    
    # Numerator: Linear combination of basis functions with full weight
    num = (p[0] * 1.0 +
           p[1] * sin_x +
           p[2] * cos_x +
           p[3] * sin_v +
           p[4] * cos_v +
           p[5] * sin_xv +
           p[6] * cos_xv +
           p[7] * sin_x_minus_v +
           p[8] * cos_x_minus_v +
           p[9] * x_sin_v)
           
    # Denominator: 1 + scaled linear combination. 
    # The scaling factor 0.1 keeps the denominator near 1 for stability 
    # but allows the ratio to deviate significantly from the numerator-only case.
    scale = 0.1
    den = (1.0 +
           p[1] * scale * sin_x +
           p[2] * scale * cos_x +
           p[3] * scale * sin_v +
           p[4] * scale * cos_v +
           p[5] * scale * sin_xv +
           p[6] * scale * cos_xv +
           p[7] * scale * sin_x_minus_v +
           p[8] * scale * cos_x_minus_v +
           p[9] * scale * x_sin_v)
           
    # Safety guard: prevent division by zero
    # If den is too close to 0, we clip it to a small epsilon with correct sign
    # However, since den starts at 1 and params start at 1, and scale is small,
    # den is likely positive. We use np.where to handle potential sign flips robustly.
    # Actually, simpler: just clip absolute value away from 0.
    # But we must preserve sign for continuity.
    # Let's just use a safe division.
    eps = 1e-6
    safe_den = np.where(np.abs(den) < eps, np.sign(den) * eps, den)
    # Handle case where den is exactly 0 (sign is 0)
    safe_den = np.where(safe_den == 0, eps, safe_den)
    
    out = num / safe_den
    
    return out
