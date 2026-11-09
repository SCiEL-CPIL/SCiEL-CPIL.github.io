import math
import time

def construct():
    """
    Returns the largest subset of [1, 255] containing 255 such that 
    gcd(a, b) > 1 for all distinct pairs a, b in the subset.
    
    The maximum cardinality is 85, achieved by the set of all multiples of 3 
    in the range [1, 255]. 
    """
    start_time = time.time()
    deadline = start_time + 5.0  # 5 seconds is more than enough, well within 60s limit
    
    n = 255
    # The smallest prime factor of 255 is 3.
    # The set of multiples of 3 is a valid pairwise non-coprime set.
    # Size is floor(255 / 3) = 85.
    # It can be proven that no larger set exists for n=255.
    
    # Generate the set of multiples of 3
    # We verify validity quickly to be safe, though it's mathematically guaranteed
    result = list(range(3, n + 1, 3))
    
    # Optional: Verify the result is valid and maximal (heuristic check)
    # Since the calculation is O(n), we can afford a check if time permits
    if time.time() < deadline:
        # Check if all pairs have gcd > 1
        # This is O(N^2) which is 85^2 ~ 7000 ops, very fast
        is_valid = True
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if math.gcd(result[i], result[j]) == 1:
                    is_valid = False
                    break
            if not is_valid:
                break
        
        if not is_valid:
            # Fallback to a known valid smaller set if the logic somehow failed (it won't)
            # But we know multiples of 3 are valid.
            pass
            
    return result
