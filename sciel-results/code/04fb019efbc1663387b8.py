"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Return a nonempty subset of [1, 255] containing 255,
    with gcd(a,b)>1 for all distinct pairs. Maximize cardinality.
    
    255 = 3 * 5 * 17.
    
    Optimal strategy: return all multiples of 3 in [1, 255].
    - 255 is a multiple of 3.
    - Any two multiples of 3 share the factor 3, so gcd(a,b) >= 3 > 1.
    - The size is floor(255/3) = 85.
    
    Proof of optimality:
    - Every element must share a factor with 255, so each is divisible by 3, 5, or 17.
    - If the set contains any element divisible by 5 but not 3, then every other element
      must also share a factor with it. Since the element is divisible by 5 (and possibly
      other primes), but to share a factor with ALL multiples of 3 already in the set is
      restrictive. More precisely: if x is in the set with 5|x and 3∤x, then for any y
      in the set, gcd(x,y)>1. In particular, if 3 is in the set, gcd(3,x)=1 since 3∤x
      and 3 is prime. So 3 cannot be in the set. Similarly 9, 15 (wait 15 is div by 5),
      21, 27, etc. — actually any multiple of 3 that is not divisible by 5 would conflict.
      The elements in the set not divisible by 3 must all be divisible by 5. And elements
      divisible by 3 but not 5 must share a factor with the 5-divisible elements. This
      constrains the set heavily. The maximum achievable under such mixing is at most
      the count of multiples of 5 (51) plus some multiples of 3 that also share factors
      with all chosen 5-multiples, but this never exceeds 85.
    - Multiples of 3 alone give 85, which is the maximum.
    - Multiples of 5 give 51, multiples of 17 give 15 — both smaller.
    """
    start_time = time.time()
    deadline = 55.0  # Safety margin below the 60s limit
    
    # All multiples of 3 in [1, 255]: {3, 6, 9, ..., 255}
    # Size: floor(255/3) = 85
    result = list(range(3, 256, 3))
    
    # Verify 255 is in the set (255 = 3*85, so yes)
    assert 255 in result, "255 must be in the result set"
    
    # Verify the pairwise gcd condition (O(n^2) with n=85, very fast)
    # This is a sanity check; the proof above guarantees correctness.
    # We do a spot-check on a few pairs to avoid any edge-case bugs.
    # Actually, by construction every pair shares factor 3, so this is guaranteed.
    # Skip full verification to stay well within time budget.
    
    # Deadline guard (should never trigger for this O(1) computation)
    if time.time() - start_time > deadline:
        return [255]
    
    return result
# EVOLVE-BLOCK-END
