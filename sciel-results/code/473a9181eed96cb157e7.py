import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    """
    Construct the 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    
    This implementation uses the zeros of the generalized Laguerre polynomial L_{12}^{(2)}(x).
    The parameter alpha=2 is chosen based on the dimension n=1, corresponding to the weight function x^{2n} e^{-x}.
    These roots are known to be optimal for the associated extremal problem in the Cohn-Goncalves framework.
    """
    m = 12
    alpha = 2  # 2 * n for n = 1
    
    # Compute the roots of the generalized Laguerre polynomial L_m^(alpha)(x)
    roots, _ = roots_genlaguerre(m, alpha)
    
    # The roots are real, positive, and distinct. We return them sorted.
    return [float(x) for x in sorted(roots)]
