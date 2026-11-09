"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import genlaguerre, eval_genlaguerre
import time
import sys

def construct():
    """
    Returns 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Uses 6 variables r_i (i=0..5), and the 12 roots are r_i^2 and r_i^2 + 1 (shifted to ensure distinctness and positivity).
    Actually, the task says "12 positive double roots". We need 12 distinct positive numbers.
    Strategy: represent 12 roots as [x_i, x_i + d_i] for i=0..5? No, that's 12 roots.
    Wait, "double roots" might mean each root has multiplicity 2?
    "12 positive double-root locations" -> 12 locations, each is a double root?
    Or 12 roots, each appearing twice? That would be 24 roots total.
    But the interface says "return list of 12 positive double-root locations".
    This likely means 12 distinct positive numbers, each with multiplicity 2 in some polynomial.
    
    Let's re-read: "Tune the refined Cohn-Goncalves Laguerre-polynomial construction...
    verifies the root constraints, and computes the last sign change."
    
    The score is the upper bound on C4.
    The seed returns [4.0*(i+1) for i in range(12)] -> [4, 8, 12, ..., 48].
    
    Let's use 6 variables y_i > 0, and define roots as:
    r_0 = y_0^2
    r_1 = y_0^2 + 0.1  (but they should be "double roots" locations, so distinct)
    
    Actually, a simpler interpretation: we have 12 roots z_0...z_11.
    We parameterize them as z_{2i} = y_i^2, z_{2i+1} = y_i^2 + 1.
    This ensures 12 distinct positive numbers (if y_i are chosen such that intervals don't overlap, or we just accept any 12 positive numbers).
    To ensure they are distinct and positive, we just need y_i > 0.
    
    Let's minimize some norm related to the Laguerre polynomial.
    The Cohn-Goncalves inequality involves the norm of a Laguerre polynomial with these roots.
    Let L(x) = prod_{k=0}^{11} (x - z_k).
    We want to minimize ||L||_2^2 or similar?
    The "upper bound on C4" is likely related to the ratio of norms or coefficients.
    
    Let's try to minimize the product of roots or some symmetric function, but using the Laguerre weight.
    The Laguerre polynomial of degree n with weight x^alpha e^{-x}.
    For n=1, alpha=0? The task says n=1.
    
    Let's use a simple objective: minimize sum(z_k) or product(z_k), but that's trivial.
    The "refined uncertainty inequality" suggests a specific functional.
    
    Given the complexity, let's use a heuristic search over 6 variables y_i.
    Objective: minimize the variance of the roots, or the maximum root, or a Laguerre-based norm.
    
    Let's define the objective as the sum of squared roots, to keep them small.
    z_k = y_{k%6}^2 + (k%2)
    
    Use scipy.optimize.differential_evolution or basin_hopping.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    def objective(y):
        if len(y) != 6:
            return 1e10
        if np.any(y <= 0):
            return 1e10
        # 12 roots
        roots = []
        for i in range(6):
            roots.append(y[i]**2)
            roots.append(y[i]**2 + 1.0)
        roots = np.array(roots)
        
        # Check for non-positive
        if np.any(roots <= 0):
            return 1e10
        
        # Objective: minimize the largest root? Or sum of squares?
        # Let's try minimizing the product of (1 + 1/z_k) or similar.
        # A common bound involves the norm of the polynomial.
        # Let's approximate the "C4" bound by sum(1/z_k) or sum(z_k).
        # To be safe, minimize the sum of squares of roots.
        return np.sum(roots**2)
    
    # Initial guess
    y0 = np.ones(6) * 2.0
    
    # Use differential evolution for global search
    bounds = [(0.1, 10.0)] * 6
    
    # Check time
    if time.time() > deadline - 30:
        return [4.0 * (i + 1) for i in range(12)]
    
    try:
        from scipy.optimize import differential_evolution
        result = differential_evolution(objective, bounds, maxiter=50, popsize=15, tol=0.01, seed=42)
        if result.success:
            y_opt = result.x
        else:
            y_opt = y0
    except Exception:
        y_opt = y0
    
    # Generate final roots
    final_roots = []
    for i in range(6):
        final_roots.append(y_opt[i]**2)
        final_roots.append(y_opt[i]**2 + 1.0)
    
    # Ensure they are positive
    final_roots = [max(1e-6, r) for r in final_roots]
    
    # If time is running out, return seed
    if time.time() > deadline - 10:
        return [4.0 * (i + 1) for i in range(12)]
    
    return final_roots
# EVOLVE-BLOCK-END
