"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model using a scaled, well-conditioned basis.
    
    Combines the Van der Pol-inspired structure of the current program (capturing
    amplitude-dependent damping via x^3, x^5, v^3 terms) with the input scaling
    strategy from inspiration 1 and 3 to improve BFGS convergence and conditioning.
    
    The inputs x and v are centered and scaled to approximately [-1, 1] based on
    the known training ranges, which helps the optimizer navigate the parameter space
    more effectively. The basis functions are chosen to capture the essential
    nonlinear dynamics of a limit-cycle oscillator while maintaining good numerical
    properties.
    
    Basis: {1, X, V, X^2, V^2, XV, X^3, V^3, X^2V, XV^2}
    where X and V are scaled versions of x and v.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Scale x and v to approximately [-1, 1] for better conditioning
    # x range: [-0.5728, 0.5275], center ~ -0.02265, half-width ~ 0.55015
    # v range: [-0.1754, 0.2821], center ~ 0.05335, half-width ~ 0.22875
    X = (x + 0.02265) / 0.55015
    V = (v - 0.05335) / 0.22875
    
    # Compute scaled polynomial basis terms
    X2 = X * X
    X3 = X2 * X
    V2 = V * V
    V3 = V2 * V
    XV = X * V
    X2V = X2 * V
    XV2 = X * V2
    
    # Combine basis terms with fitted parameters
    # This captures: constant, linear x, linear v, quadratic x, 
    # cross term xv, quadratic v, cubic x, x^2 v, x v^2, cubic v
    return (p0 + 
            p1 * X + 
            p2 * V + 
            p3 * X2 + 
            p4 * V2 + 
            p5 * XV + 
            p6 * X3 + 
            p7 * V3 + 
            p8 * X2V + 
            p9 * XV2)
# EVOLVE-BLOCK-END
