"""Rewritten program for symbolic regression of a nonlinear oscillator."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression for a nonlinear oscillator acceleration.
    
    Uses a mixed basis of:
    - Polynomial terms in x and v (up to cubic)
    - Coupling terms (x*v, x^2*v, x*v^2)
    - Periodic drive terms sin(t), cos(t)
    - Parametric coupling terms sin(t)*x, cos(t)*v
    
    10 parameters total.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Basis functions
    b1 = np.ones_like(t)           # bias
    b2 = np.sin(t)                 # periodic drive
    b3 = np.cos(t)                 # periodic drive phase
    b4 = x                         # linear restoring
    b5 = v                         # linear damping
    b6 = x * x                     # quadratic stiffness
    b7 = x * v                     # velocity-dependent damping
    b8 = np.sin(t) * x             # parametric excitation
    b9 = np.cos(t) * v             # parametric damping
    b10 = x * x * x                # cubic nonlinearity (Duffing)
    
    # Linear combination
    return (params[0] * b1 + 
            params[1] * b2 + 
            params[2] * b3 + 
            params[3] * b4 + 
            params[4] * b5 + 
            params[5] * b6 + 
            params[6] * b7 + 
            params[7] * b8 + 
            params[8] * b9 + 
            params[9] * b10)

# EVOLVE-BLOCK-END
