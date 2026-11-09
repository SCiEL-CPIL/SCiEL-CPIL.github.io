import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    # Task: m=12, n=1.
    # Directive: Use exact zeros of generalized Laguerre polynomial L_12^(1)(x).
    # scipy.special.roots_genlaguerre(N, alpha) computes roots of L_N^(alpha)(x).
    # Here N=12, alpha=1.
    roots, _ = roots_genlaguerre(12, 1)
    
    # Return the 12 positive root locations as a list of floats.
    return [float(r) for r in roots]
