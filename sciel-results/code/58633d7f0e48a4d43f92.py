import math
import time

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, 
    with gcd(a, b) > 1 for all distinct pairs, maximizing cardinality.
    
    The optimal strategy for n=255 (3*5*17) is to take all multiples of the 
    smallest prime factor, which is 3.
    Size = 255 // 3 = 85.
    """
    # Deadline guard, though this calculation is instantaneous
    start_time = time.time()
    deadline = 90.0
    
    # The set of all multiples of 3 in [1, 255]
    # This set contains 255 (since 255 % 3 == 0)
    # For any a, b in this set, 3 divides a and 3 divides b, so gcd(a, b) >= 3 > 1.
    result = [x for x in range(1, 256) if x % 3 == 0]
    
    # Optional: Verification step to ensure validity
    # This is O(N^2) but N=85, so it's very fast.
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) == 1:
                # Fallback if logic fails (should not happen)
                return [255]
        
    if time.time() > deadline:
        return result
        
    return result
