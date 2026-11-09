import numpy as np
from scipy.special import roots_genlaguerre
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 500.0  # Safety margin

    # Strategy: Use the exact zeros of the generalized Laguerre polynomial
    # L_12^{(1)}(x) as the root locations.
    # The task specifies m=12, n=1.
    # For n=1, the weight function is typically x^1 e^{-x} or related.
    # The generalized Laguerre polynomial L_n^{(\alpha)}(x) has orthogonality
    # with weight x^\alpha e^{-x}.
    # Here, we hypothesize that the optimal roots for the uncertainty bound
    # correspond to the roots of L_m^{(\alpha)} where alpha is related to n.
    # Given n=1, alpha is likely 1 (or potentially n-1=0, or n+1=2).
    # Let's try alpha=1 first, as it's the most direct association with n=1
    # in the context of Laguerre polynomials often used in such bounds.
    # If the "dimension n=1" implies the parameter alpha for the Laguerre
    # polynomial L_m^{(\alpha)}, then alpha=1 is a strong candidate.
    
    # Let's check the inspiration: "Use the exact zeros of the generalized
    # Laguerre polynomial L_12^{(1)}(x)". This explicitly suggests alpha=1.
    
    m = 12
    alpha = 1.0
    
    # Get the roots of L_12^{(1)}(x)
    roots, weights = roots_genlaguerre(m, alpha)
    
    # The roots are positive and distinct.
    # Ensure they are sorted (roots_genlaguerre returns them sorted by default)
    best_x = np.sort(roots)
    
    # Verify positivity
    if not np.all(best_x > 0):
        # Fallback if something goes wrong, though roots_genlaguerre
        # guarantees positive roots for alpha > -1
        best_x = np.array([4.0 * (i + 1) for i in range(12)])
        
    return [float(x) for x in best_x]
