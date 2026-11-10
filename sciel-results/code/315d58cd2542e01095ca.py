"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Return a nonempty subset of [1,255] containing 255,
    with gcd(a,b)>1 for all distinct pairs. Maximize cardinality.
    
    The optimal strategy is to return all multiples of 3 in [1, 255].
    - 255 is a multiple of 3.
    - Any two multiples of 3 share the factor 3, so gcd(a,b) >= 3 > 1.
    - The size is floor(255/3) = 85.
    
    Analysis of other strategies:
    - Multiples of 5: Size 51.
    - Multiples of 17: Size 15.
    - Even numbers sharing a factor with 255: Size 64.
    - Other mixed sets are smaller due to connectivity constraints (e.g., odd multiples of 3 like 9 force all other elements to be multiples of 3).
    
    Therefore, 85 is the maximum cardinality.
    """
    start_time = time.time()
    deadline = 60.0
    
    # The set of all multiples of 3 in [1, 255]
    result = list(range(3, 256, 3))
    
    # Verify 255 is in the set
    if 255 not in result:
        result.append(255)
        
    # The calculation is instantaneous, but we adhere to the deadline guard pattern.
    if time.time() - start_time > deadline:
        # Fallback to a known valid small set if we somehow timed out (unlikely)
        return [255]
        
    return result
# EVOLVE-BLOCK-END
