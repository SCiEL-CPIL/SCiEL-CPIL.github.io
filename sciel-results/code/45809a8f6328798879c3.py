import math
import time

def construct():
    # The problem asks for the largest subset of [1, 255] containing 255
    # such that gcd(a, b) > 1 for all distinct pairs.
    # 
    # Analysis:
    # 1. 255 = 3 * 5 * 17.
    # 2. Any element x in the set must satisfy gcd(x, 255) > 1, so x is divisible by 3, 5, or 17.
    # 3. The set of all multiples of 3 in [1, 255] is a clique because any two multiples of 3 share a factor of 3.
    #    Size = floor(255 / 3) = 85.
    # 4. The set of all multiples of 5 has size floor(255 / 5) = 51.
    # 5. The set of all multiples of 17 has size floor(255 / 17) = 15.
    # 6. Can we form a larger clique by mixing?
    #    If the clique contains a power of 3 (3, 9, 27, 81), then all other elements must be divisible by 3.
    #    In this case, the max size is 85.
    #    If we exclude all powers of 3, we have 81 elements.
    #    To add an element y not divisible by 3, y must share a factor with all remaining elements.
    #    The remaining elements include 3*p for many primes p.
    #    This forces y to be divisible by many primes, which is impossible for y <= 255.
    #    Specifically, removing enough 3*p to allow y results in a net loss in cardinality.
    #    Thus, 85 is the maximum cardinality.
    
    # Construct the set of multiples of 3
    result = [x for x in range(1, 256) if x % 3 == 0]
    
    # 255 is a multiple of 3, so it is included.
    # Verify 255 is in result
    if 255 not in result:
        result.append(255)
        
    return result
