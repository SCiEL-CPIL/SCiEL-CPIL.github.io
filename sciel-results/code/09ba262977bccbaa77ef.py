"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression for acceleration of a nonlinear oscillator.
    
    Strategy: Physics-informed sparse basis.
    Instead of a generic polynomial expansion which suffers from multicollinearity 
    and wastes degrees of freedom on likely-zero coefficients (e.g., constant offset, 
    sin(t) if drive is cos), we select the 10 most physically relevant terms 
    for a driven, damped, nonlinear oscillator (Duffing/Van der Pol family).
    
    Basis terms:
    0: x          (Linear restoring force)
    1: x^3        (Cubic restoring force, Duffing)
    2: x^5        (Higher order nonlinearity, captures stiffening/softening beyond cubic)
    3: v          (Linear damping)
    4: x^2*v      (Nonlinear damping, Van der Pol)
    5: x*v        (Coupling term)
    6: v^2        (Quadratic drag)
    7: cos(t)     (Primary periodic drive)
    8: x*cos(t)   (Parametric resonance / modulation)
    9: x^2*cos(t) (Nonlinear parametric modulation)
    
    This structure allows the optimizer to find the specific Duffing-Van der Pol 
    parameters directly without interference from irrelevant basis vectors.
    """
    p = params
    
    # Pre-compute powers to ensure numerical stability and speed
    x2 = x * x
    x3 = x2 * x
    x5 = x3 * x2
    
    # Features based on physics of nonlinear oscillators
    f0 = x
    f1 = x3
    f2 = x5
    f3 = v
    f4 = x2 * v
    f5 = x * v
    f6 = v * v
    f7 = np.cos(t)
    f8 = x * np.cos(t)
    f9 = x2 * np.cos(t)
    
    # Linear combination
    y = (p[0] * f0 + 
         p[1] * f1 + 
         p[2] * f2 + 
         p[3] * f3 + 
         p[4] * f4 + 
         p[5] * f5 + 
         p[6] * f6 + 
         p[7] * f7 + 
         p[8] * f8 + 
         p[9] * f9)
         
    return y

# EVOLVE-BLOCK-END
