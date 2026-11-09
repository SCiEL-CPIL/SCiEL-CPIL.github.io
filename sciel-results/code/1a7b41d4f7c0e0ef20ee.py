"""
Constructs a maximum-cardinality subset of [1, 255] containing 255
where every distinct pair has gcd > 1.

255 = 3 * 5 * 17. The optimal strategy is to take all multiples of the
smallest prime factor (3), yielding 85 elements. Mixing primes from different
factor groups creates coprime pairs, so a single-prime-factor set is optimal.

This program keeps the direct constructive core (multiples of 3) and adds a
lightweight verification pass as a safety net.
"""

import time
from math import gcd

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, with gcd(a,b) > 1
    for all distinct pairs, maximizing cardinality.
    """
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    # All multiples of 3 in [1, 255]: size = 85
    # This is valid because gcd(3a, 3b) >= 3 > 1 for all a, b.
    result = list(range(3, 256, 3))

    # 255 = 3*85, so it's already in the list
    # Verify 255 is present
    if 255 not in result:
        result.append(255)

    # Quick verification (85 elements, ~3570 pairs, very fast)
    if time.time() - start_time < deadline:
        n = len(result)
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                if gcd(result[i], result[j]) == 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            return [255]

    return result
