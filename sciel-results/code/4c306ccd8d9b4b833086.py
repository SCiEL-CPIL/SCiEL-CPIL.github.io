import math
import time

def construct():
    # The task requires a subset of [1, 255] containing 255,
    # with gcd(a, b) > 1 for all distinct pairs, maximizing cardinality.
    
    # 255 = 3 * 5 * 17.
    # Any element in the set must share a factor with 255, so it must be divisible by 3, 5, or 17.
    # If the set contains any number not divisible by 3, say x, then x must be divisible by 5 or 17.
    # If x is divisible by 5 (and not 3), then all other elements y must satisfy gcd(x, y) > 1.
    # Since x is not divisible by 3, y cannot rely on the factor 3 to share a gcd with x.
    # If we include 3 in the set, then for any x not divisible by 3, gcd(3, x) must be > 1, which implies 3|x, a contradiction.
    # Thus, if 3 is in the set, all elements must be divisible by 3.
    # The set of all multiples of 3 in [1, 255] has size floor(255/3) = 85.
    
    # Can we get a larger set by excluding 3?
    # If we exclude all numbers that are "exclusive" to one prime factor (e.g., divisible by 3 but not 5 or 17),
    # the set size drops significantly because we are forced into intersections like multiples of 15, 51, 85, etc.
    # The set of multiples of 3 is a clique.
    # The set of multiples of 5 is a clique of size 51.
    # The set of multiples of 17 is a clique of size 15.
    # Mixing them reduces the size due to compatibility constraints.
    # 85 is the maximum.
    
    return list(range(3, 256, 3))
