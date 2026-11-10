"""Program to find a large pairwise non-coprime subset of [1,255] containing 255."""
import math
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Return a nonempty subset of [1,255] containing 255,
    with gcd(a,b)>1 for all distinct pairs. Maximize cardinality.
    
    Key insight: 3 is in [1,255] and is prime. Any element x in a pairwise
    non-coprime set containing 3 must satisfy gcd(x,3)>1, so x must be
    divisible by 3. There are exactly floor(255/3)=85 multiples of 3 in
    [1,255], and they all share factor 3, forming a valid clique of size 85.
    This is the theoretical maximum.
    """
    # All multiples of 3 in [1,255] form a pairwise non-coprime set (all share factor 3)
    # 255 = 3*85, so 255 is included
    result = list(range(3, 256, 3))
    return result
# EVOLVE-BLOCK-END
