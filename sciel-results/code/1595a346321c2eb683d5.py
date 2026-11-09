"""Refined Cohn-Goncalves Laguerre construction for Fourier sign uncertainty."""
import numpy as np
from scipy.special import roots_genlaguerre
import time

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    deadline = time.time() + 600.0
    
    best_roots = None
    best_product = -1.0
    
    # Search for the generalized Laguerre polynomial L_12^alpha(x)
    # that maximizes the product of its roots.
    # Based on inspiration, score ~ 1/sqrt(product), so max product minimizes score.
    # The seed uses a grid, which has a large product.
    # L_12^0 has product 12!.
    # L_12^alpha has product 12! * C(12+alpha, 12).
    # We want to maximize this.
    
    # We assume alpha can be tuned.
    # Let's try a range of alphas.
    # If alpha is an integer, C(12+alpha, 12) is well-defined.
    # For non-integer alpha, it's a generalized binomial coefficient.
    # roots_genlaguerre works for real alpha > -1.
    
    # Let's sample alpha values.
    # The seed product corresponds to alpha ~ 8.
    # We want alpha > 8.
    # Let's try up to 50.
    
    for alpha in range(0, 51):
        if time.time() > deadline - 10.0:
            break
        try:
            r, _ = roots_genlaguerre(12, alpha)
            root_list = sorted([float(x) for x in r])
            if len(root_list) == 12 and all(x > 0 for x in root_list):
                prod = np.prod(root_list)
                if prod > best_product:
                    best_product = prod
                    best_roots = root_list
        except Exception:
            continue
            
    # If no roots found, fallback
    if best_roots is None:
        # Fallback to standard L_12
        try:
            r, _ = roots_genlaguerre(12, 0)
            best_roots = sorted([float(x) for x in r])
        except Exception:
            best_roots = [4.0 * (i + 1) for i in range(12)]
            
    return best_roots
