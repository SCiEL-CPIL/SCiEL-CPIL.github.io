import time
from math import gcd

def construct():
    """
    Returns the maximum cardinality subset of [1, 255] containing 255 
    such that gcd(a, b) > 1 for all distinct pairs a, b in the subset.
    
    Analysis:
    255 = 3 * 5 * 17.
    Any element x in the set must have gcd(x, 255) > 1, so x is divisible by 3, 5, or 17.
    The set of all multiples of 3 in [1, 255] is a valid clique because any two 
    multiples of 3 share the factor 3.
    Size of multiples of 3: floor(255/3) = 85.
    
    Can we do better?
    If the set contains 3, then for any other element y, gcd(3, y) > 1 implies 3 divides y.
    Thus, if 3 is in the set, the set is a subset of multiples of 3. Max size 85.
    
    If 3 is NOT in the set, we might mix multiples of 3, 5, and 17.
    However, any element y not divisible by 3 must be compatible with ALL elements 
    in the set that ARE divisible by 3.
    Let A be the subset of multiples of 3 in our solution.
    Let B be the subset of non-multiples of 3 in our solution.
    For any b in B and a in A, gcd(a, b) > 1.
    Since 3 does not divide b, the common factor must be 5 or 17 (or other primes, 
    but a is a multiple of 3, so a's prime factors are {3} U P_a).
    If B contains an element b that is not divisible by 17, then for all a in A, 
    gcd(a, b) > 1 implies they share a factor p != 3, 17. 
    If b is only divisible by 5 (e.g., 5, 25, 35...), then all a in A must be 
    divisible by 5. 
    The number of multiples of 15 in [1, 255] is 17.
    So |A| <= 17.
    The max size of B (multiples of 5 not div by 3) is 34.
    Total size <= 17 + 34 = 51 < 85.
    
    Similar arguments hold for other combinations. The set of multiples of 3 is optimal.
    """
    # Generate all multiples of 3 up to 255
    result = list(range(3, 256, 3))
    
    # Verify constraints (optional but good for safety)
    assert 255 in result
    assert all(1 <= x <= 255 for x in result)
    # Verify pairwise gcd > 1
    # Since all are multiples of 3, gcd is at least 3.
    # This check is O(N^2) with N=85, so ~7000 ops, very fast.
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) <= 1:
                # This should never happen
                raise ValueError("Invalid set generated")
                
    return result
