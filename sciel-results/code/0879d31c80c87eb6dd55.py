import time
from math import gcd

def construct():
    """
    Return a nonempty subset of [1, 255] containing 255, with gcd(a,b)>1 for all distinct pairs.
    Maximize cardinality.
    """
    # The optimal strategy for maximizing the size of a pairwise non-coprime subset
    # of [1, N] containing N (where N is square-free with prime factors p1 < p2 < ... < pk)
    # is to take all multiples of the smallest prime factor p1.
    # For N = 255 = 3 * 5 * 17, the smallest prime factor is 3.
    # The set of multiples of 3 in [1, 255] has size 255 // 3 = 85.
    # This set is a clique because any two multiples of 3 share the factor 3.
    # It is known that no larger clique exists for this N.
    
    # We explicitly construct this set.
    # 255 is divisible by 3, so it is included.
    result = [i for i in range(3, 256, 3)]
    
    return result
