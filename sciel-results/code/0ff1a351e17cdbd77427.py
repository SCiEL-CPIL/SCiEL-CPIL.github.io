import math
import time

def construct():
    """
    Return a nonempty subset of [1,255] containing 255, with gcd(a,b)>1 for
    all distinct pairs. Maximize cardinality.
    
    255 = 3 * 5 * 17.
    The set of all multiples of 3 in [1,255] has size floor(255/3) = 85.
    This is optimal: any valid set containing 255 must have every element
    share a factor with 255 (i.e., be divisible by 3, 5, or 17). To maximize
    size, we pick the prime factor with the most multiples, which is 3.
    """
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill

    n = 255
    # 255 = 3 * 5 * 17
    # Multiples of 3 in [1, 255]: 3, 6, 9, ..., 255 → count = 255 // 3 = 85
    # All pairs share factor 3, so gcd ≥ 3 > 1. ✓
    # 255 is included since 255 % 3 == 0. ✓
    
    if time.time() > deadline:
        return [255]
    
    return list(range(3, n + 1, 3))
