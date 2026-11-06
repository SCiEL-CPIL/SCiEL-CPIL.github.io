import math

def construct():
    # Target: Maximize size of subset S of [1, 255] containing 255
    # such that gcd(a, b) > 1 for all distinct a, b in S.
    
    # 255 = 3 * 5 * 17
    # The smallest prime factor is 3.
    # The set of all multiples of 3 in [1, 255] has size floor(255/3) = 85.
    # Any other number x in [1, 255] not divisible by 3 must share a factor > 1 with all elements in S.
    # Since S contains 3 (or 6, 9, etc.), x must share a factor with 3.
    # If x is not divisible by 3, gcd(x, 3) = 1, so x cannot be in S if 3 is in S.
    # To maximize size, we stick to the multiples of 3.
    
    max_val = 255
    candidate_set = set()
    
    # Include 255 as required
    candidate_set.add(255)
    
    # Add all other multiples of 3
    for i in range(1, max_val + 1):
        if i % 3 == 0 and i != 255:
            candidate_set.add(i)
            
    # Convert to list for return
    result = list(candidate_set)
    
    # Ensure 255 is present (it is by construction, but good for safety)
    if 255 not in result:
        result.append(255)
        
    return result
