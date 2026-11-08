"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model with coupled terms.
    
    This strategy moves away from purely additive polynomials to include
    interaction terms (couplings) between time, position, and velocity.
    This is critical for oscillators where:
    1. Drive may be parametric (t * x) or modulated.
    2. Damping may be velocity-dependent (v * x or v * v).
    3. Nonlinear stiffness may depend on time (t * x^2).
    
    Parameters:
    t: time
    x: position
    v: velocity
    params: array of 10 fitted constants
    
    Basis functions (10 terms):
    1. 1
    2. x
    3. v
    4. x^2
    5. x^3
    6. v^2
    7. x * v  (coupled state nonlinearity)
    8. t_c * x (parametric drive)
    9. t_c * v (time-dependent damping)
    10. t_c^2 (pure time drive quadratic)
    """
    # Center and scale time to improve conditioning
    # Range [30, 50] -> center 40, scale 10
    tc = (t - 40.0) / 10.0
    tc2 = tc * tc
    
    # State variables powers
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    
    # Coupled terms
    xv = x * v
    tx = tc * x
    tv = tc * v

    res = (params[0] + 
           params[1] * x + 
           params[2] * v + 
           params[3] * x2 + 
           params[4] * x3 + 
           params[5] * v2 + 
           params[6] * xv + 
           params[7] * tx + 
           params[8] * tv + 
           params[9] * tc2)
           
    return res
# EVOLVE-BLOCK-END
