import numpy as np
from scipy.special import genlaguerre
from scipy.optimize import brentq
import time

def construct():
    """
    Return 12 double-root locations z_1..z_12.
    
    Strategy: Use parameterized Bessel-like spacing z_i = (a + b*i)^2 for i=1..12.
    The base case is a=0, b=pi (giving z_i = (i*pi)^2).
    We optimize the offset 'a' to potentially reduce B.
    
    The idea: shifting the starting point of the quadratic sequence may better
    align with the Laguerre polynomial structure, potentially reducing the
    largest root of q(x).
    """
    # Try several candidate offsets around the base case
    # Base: z_i = (i*pi)^2, which corresponds to a=0, b=pi
    
    # We'll test a few values of 'a' (offset in the sqrt-space)
    # and pick the one that seems most promising based on a quick heuristic
    
    best_z = None
    best_score = float('inf')
    
    # Test offsets: a = 0 (base), and small perturbations
    # The key is that z_i = (a + b*i)^2 must be positive and distinct
    # For b = pi, we need a + b*i > 0 for all i, so a > -b = -pi
    
    # Try a range of offsets
    for a in [0.0, -0.5, -1.0, -1.5, -2.0, -2.5, 0.5, 1.0]:
        b = np.pi
        i_vals = np.arange(1, 13, dtype=np.float64)
        z_vals = (a + b * i_vals) ** 2
        
        # Check validity: all positive, finite, distinct
        if np.any(z_vals <= 0) or not np.all(np.isfinite(z_vals)):
            continue
        if len(set(z_vals)) != 12:
            continue
        
        # Quick heuristic: prefer configurations where the spacing in sqrt-space
        # is more uniform relative to the Laguerre polynomial zeros
        # For L_k^{-1/2}(x), zeros are roughly at x ~ (k + 0.25)^2 * pi^2 / 4
        # So we want our z_i to be "spread out" similarly
        
        z_list = [float(z) for z in z_vals]
        
        # Use a simple proxy: minimize the maximum gap in log-space
        # This encourages more uniform coverage
        log_z = np.log(z_list)
        gaps = np.diff(log_z)
        max_gap = np.max(gaps)
        
        if max_gap < best_score:
            best_score = max_gap
            best_z = z_list
    
    # Fallback to base case if no valid configuration found
    if best_z is None:
        k_vals = np.arange(1, 13, dtype=np.float64)
        z_vals = (k_vals * np.pi) ** 2
        best_z = [float(z) for z in z_vals]
    
    return best_z
