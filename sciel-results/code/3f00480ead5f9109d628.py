"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration with damping, nonlinear stiffness, 
    and time-dependent driving force.
    
    Uses up to 10 fitted constants to model:
    - Linear and nonlinear terms in x (cubic, quintic for softening/hardening)
    - Linear and nonlinear damping in v
    - Time-dependent driving force (sinusoidal with phase/amplitude modulation)
    - Constant offset
    
    The general form is:
    a = p0*x + p1*x^3 + p2*x^5 + p3*v + p4*v^3 + p5*sin(p6*t + p7*x) + p8*cos(p9*t) + p10
    
    But we only have 10 params, so let's structure it carefully:
    p0: linear stiffness
    p1: cubic stiffness (nonlinearity)
    p2: quintic stiffness (higher-order nonlinearity)
    p3: linear damping
    p4: cubic damping (nonlinear damping)
    p5: driving amplitude
    p6: driving frequency
    p7: driving phase
    p8: constant offset
    p9: cross-term coefficient (x*v interaction)
    """
    p = params
    
    # Nonlinear spring force terms
    spring = p[0] * x + p[1] * x**3 + p[2] * x**5
    
    # Nonlinear damping terms
    damping = p[3] * v + p[4] * v**3
    
    # Time-dependent driving force: sinusoidal with frequency and phase
    drive = p[5] * np.sin(p[6] * t + p[7])
    
    # Cross-term: velocity-dependent stiffness modulation
    cross = p[8] * x * v
    
    # Constant offset
    offset = p[9]
    
    return spring + damping + drive + cross + offset
# EVOLVE-BLOCK-END
