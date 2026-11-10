"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Return a nonempty subset of [1,255] containing 255,
    with gcd(a,b)>1 for all distinct pairs. Maximize cardinality.
    
    255 = 3 * 5 * 17. The largest pairwise noncoprime set containing 255
    is all multiples of 3 in [1,255], giving 85 elements.
    
    Proof of optimality:
    - Any set where all pairs share factor 3: at most 85 elements (multiples of 3).
    - Any set where all pairs share factor 5: at most 51 elements.
    - Any set where all pairs share factor 17: at most 15 elements.
    - Mixed sets: If an element x in the set is not divisible by 5 or 17,
      then gcd(255,x) > 1 requires x divisible by 3, and every other element
      y must have gcd(x,y) > 1. Since x may only share factor 3, all y must
      be divisible by 3. Thus the set is contained in multiples of 3.
    - If all elements are divisible by 5 or 17 (but not necessarily 3), the
      set is bounded by multiples of 5 (51) or 17 (15), both < 85.
    
    Therefore 85 is the maximum cardinality.
    """
    # All multiples of 3 in [1, 255]: 3, 6, 9, ..., 255
    # Count: 255 // 3 = 85
    return list(range(3, 256, 3))
# EVOLVE-BLOCK-END
