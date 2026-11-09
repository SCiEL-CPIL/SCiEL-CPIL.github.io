import numpy as np
import time
from numpy.polynomial.laguerre import laggauss

def construct():
    """
    Return a list of 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    The score is the upper bound on C4 (lower is better).
    """
    t0 = time.time()
    deadline = 620.0  # safety margin before 630s kill
    
    m = 12
    
    # Get the zeros of the Laguerre polynomial L_m(x)
    # laggauss(m) returns points (zeros) and weights
    points, weights = laggauss(m)
    points = np.sort(points)
    
    # The current best known score (33.17) came from arithmetic progression [4, 8, ..., 48] scaled by 10.
    # This corresponds to roots [40, 80, ..., 480].
    # The Laguerre zeros for m=12 range from approx 0.4 to 45.
    # Scaling by 10 gives range [4, 450].
    # This seems like a reasonable scale factor to start with.
    
    # Strategy: Model roots as s * zeros(L_m).
    # We suspect the optimal scale s is around 10 based on the arithmetic progression success.
    # Let's try a few scales and pick the one that seems most "central" or just stick with 10.
    # Since we cannot compute the true score locally, we rely on the heuristic that 
    # the orthogonal polynomial zeros are the "natural" optimal nodes for Laguerre-weighted problems.
    
    # If the problem is to minimize a bound involving the Laguerre weight e^{-x},
    # the unscaled zeros (s=1) might be optimal if the weight is fixed.
    # However, the arithmetic progression success at s=10 suggests that the "effective" scale
    # in the problem might be larger, or the bound decreases with scale up to a point.
    
    # Let's assume the scaling factor s=10 is a good candidate.
    # We will return the Laguerre zeros scaled by 10.
    
    scale = 10.0
    roots = [float(p) * scale for p in points]
    
    # Check time
    if time.time() - t0 > deadline:
        return roots
        
    return roots
