"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Return a nonempty subset of [1,255] containing 255,
    with gcd(a,b)>1 for all distinct pairs. Maximize cardinality.
    
    255 = 3 * 5 * 17.
    
    Strategy:
    The optimal set is all multiples of 3 in [1, 255].
    - 255 is a multiple of 3.
    - Any two multiples of 3 share the factor 3, so gcd(a,b) >= 3 > 1.
    - The size is floor(255/3) = 85.
    
    Why this is optimal:
    - Every element must share a factor with 255, so each element is divisible by 3, 5, or 17.
    - Multiples of 3: 85 elements.
    - Multiples of 5: 51 elements.
    - Multiples of 17: 15 elements.
    - Mixing primes yields a smaller set because adding a number from a different prime group
      (e.g., a multiple of 5 not divisible by 3) forces the removal of many numbers from the
      larger group (multiples of 3) that are coprime to it. The loss outweighs the gain.
    - 85 is the maximum cardinality.
    """
    start_time = time.time()
    deadline = 60.0
    
    # The set of all multiples of 3 in [1, 255]
    # This is the largest clique of numbers in [1, 255] that includes 255.
    result = list(range(3, 256, 3))
    
    # Verify 255 is in the set (it is, since 255 % 3 == 0)
    if 255 not in result:
        result.append(255)
    
    # Deadline guard
    if time.time() - start_time > deadline:
        return [255]
    
    return result
# EVOLVE-BLOCK-END
