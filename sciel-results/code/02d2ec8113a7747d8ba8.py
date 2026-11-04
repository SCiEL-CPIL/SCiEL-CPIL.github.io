# EVOLVE-BLOCK-START
def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Returns a list of 12 positive root locations.
    Score is the upper bound on C4 (lower is better for the evaluator's raw value,
    but the search score is seed/value, so higher search score = better).
    """
    import numpy as np
    from scipy.special import roots_genlaguerre

    m = 12
    alpha = 1
    
    # The optimal roots for maximizing the Laguerre norm of a monic polynomial
    # of degree m are the roots of the Laguerre polynomial L_m^{(alpha)}(x).
    # This minimizes the lower-order coefficients in the Laguerre expansion,
    # thus maximizing the norm squared (since the leading coeff is fixed).
    
    # roots_genlaguerre(n, alpha) returns the roots and weights of L_n^{(alpha)}
    roots, weights = roots_genlaguerre(m, alpha)
    
    # The roots are real and positive for alpha > -1
    # Ensure they are returned as a list of floats
    return list(roots)
# EVOLVE-BLOCK-END
