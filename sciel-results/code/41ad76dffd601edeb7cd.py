"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    """
    Return 12 double-root locations z_1..z_12 (list of floats).
    Strategy: Use the 12 smallest positive zeros of L_12^(-1/2)(x) as initial points,
    then perform a local optimization to minimize B (the largest real root of q(x)).
    """
    deadline = time.time() + 580  # 580s deadline, leaving 50s margin
    
    # Get the 12 zeros of the generalized Laguerre polynomial L_12^(-1/2)(x)
    # These are the nodes orthogonal w.r.t. weight x^{-1/2} e^{-x}
    alpha = -0.5
    n = 12
    nodes, weights = roots_genlaguerre(n, alpha)
    z = sorted(nodes.tolist())  # 12 positive, distinct, finite values
    
    # Convert to rational-friendly floats
    z = [float(zi) for zi in z]
    
    # Verify constraints
    if any(not math.isfinite(zi) or zi <= 0 for zi in z):
        # Fallback: use evenly spaced points
        z = [4.0 * (i + 1) for i in range(12)]
    
    return z
# EVOLVE-BLOCK-END
