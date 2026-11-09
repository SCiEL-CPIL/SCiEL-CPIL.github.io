"""
Rewritten program to find the maximum cardinality subset of [1, 255] containing 255
such that all pairs have gcd > 1.

Optimality Analysis:
1.  Let S be a valid subset of [1, 255] containing 255.
2.  For any x in S, gcd(x, 255) must be > 1.
    255 = 3 * 5 * 17.
    Thus, x must be divisible by 3, 5, or 17.
    Let U = {x in [1, 255] | gcd(x, 255) > 1}.
    S must be a subset of U.
3.  We want to maximize |S| subject to gcd(a, b) > 1 for all a, b in S.
4.  Consider the sets of multiples of the prime factors:
    C3 = {x in U | 3 divides x}, size 85.
    C5 = {x in U | 5 divides x}, size 51.
    C17 = {x in U | 17 divides x}, size 15.
5.  The set C3 is valid because any two elements share the factor 3.
6.  Can we construct a larger set?
    Any x in U not in C3 is divisible by 5 or 17 (but not 3).
    If we add such an x to C3, we must remove all y in C3 such that gcd(x, y) = 1.
    Since x is not divisible by 3, for gcd(x, 3k) to be > 1, x must share a factor with k.
    The set C3 contains 3k for k=1..85.
    If x is fixed (e.g., x=5), it is coprime to 3k unless 5 divides k.
    The number of k in [1, 85] not divisible by 5 is 85 - 17 = 68.
    So adding x=5 requires removing 68 elements. Net change: +1 - 68 = -67.
    This logic generalizes: adding elements from outside the smallest prime's multiple set forces the removal of a large fraction of that set, reducing the total cardinality.
    Formally, the maximum size is floor(255 / min_prime_factor(255)) = 85.

Therefore, the optimal strategy is to return all multiples of 3.
"""

import math
import time

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, with gcd(a,b) > 1
    for all distinct pairs, maximizing cardinality.
    """
    start_time = time.time()
    
    # The prime factors of 255 are 3, 5, 17.
    # The maximum cardinality set is the set of multiples of the smallest prime factor, 3.
    # Size: 255 // 3 = 85.
    
    # Construct the set of multiples of 3 in [1, 255]
    # range(3, 256, 3) yields 3, 6, ..., 255
    result = list(range(3, 256, 3))
    
    # 255 is a multiple of 3, so it is included.
    # Verification: 255 is in result.
    
    # Safety check for time (though this is instantaneous)
    if time.time() - start_time > 85:
        # Fallback to a minimal valid set if somehow slow (should not happen)
        return [255]
        
    # Optional: Verify the condition to be absolutely sure.
    # n=85, pairs ~ 3570. This is very fast.
    valid = True
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if math.gcd(result[i], result[j]) == 1:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # This should never happen for multiples of 3, but handle gracefully.
        # Fallback to multiples of 5 (size 51) or just {255}
        return [255]

    return result
