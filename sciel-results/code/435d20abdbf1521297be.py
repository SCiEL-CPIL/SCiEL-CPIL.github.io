"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    """
    Return a nonempty subset of [1, 255] containing 255,
    with gcd(a,b)>1 for all distinct pairs. Maximize cardinality.
    
    255 = 3 * 5 * 17.
    
    Optimal: all multiples of 3 in [1, 255], size = floor(255/3) = 85.
    - 255 is a multiple of 3.
    - Any two multiples of 3 share factor 3, so gcd >= 3 > 1.
    - No larger set exists: any element not divisible by 3 must be divisible
      by 5 or 17 to share a factor with 255, but then it's coprime to 3,
      forcing 3 out of the set and cascading constraints that keep size <= 85.
    """
    start_time = time.time()
    deadline = 55.0  # Safety margin below the 60s limit
    
    # All multiples of 3 in [1, 255]: {3, 6, 9, ..., 255}
    # Size: floor(255/3) = 85
    result = list(range(3, 256, 3))
    
    # Verify 255 is in the set
    assert 255 in result, "255 must be in the result set"
    
    # Verify pairwise gcd condition (O(n^2) with n=85, ~3600 checks, very fast)
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) <= 1:
                # Should never happen; fall back to safe answer
                return [255]
    
    # Deadline guard
    if time.time() - start_time > deadline:
        return [255]
    
    return result
# EVOLVE-BLOCK-END
