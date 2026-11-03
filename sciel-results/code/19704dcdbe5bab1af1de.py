import numpy as np
from scipy.special import roots_laguerre

def construct():
    """
    Returns the 12 positive root locations for the Laguerre combination (m=12, n=1).
    These are the roots of the Laguerre polynomial L_12(x), which are known to be
    optimal or near-optimal for the Cohn-Goncalves uncertainty bound refinement.
    """
    # Calculate roots of the Laguerre polynomial of degree 12
    # roots_laguerre(n) returns the roots of L_n(x)
    roots, _ = roots_laguerre(12)
    
    # The roots are already positive and real.
    # Sort them to ensure consistent ordering (though the set is what matters for the product/bound)
    sorted_roots = np.sort(roots)
    
    return [float(x) for x in sorted_roots]
