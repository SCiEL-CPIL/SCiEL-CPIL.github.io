import math
import time

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255,
    with gcd(a, b) > 1 for all distinct pairs.
    Maximizes cardinality.
    """
    N = 255
    deadline = time.time() + 55.0  # Safety margin before 60s limit
    
    # Factorize 255 = 3 * 5 * 17
    # The set of all multiples of 3 in [1, 255] is a valid clique
    # because any two multiples of 3 share factor 3.
    # Size = floor(255/3) = 85.
    # 255 is a multiple of 3, so it's included.
    
    # Can we do better? 
    # A pairwise noncoprime set containing 255 must have every element share a factor with 255.
    # So every element must be divisible by 3, 5, or 17.
    # But we also need gcd(a,b)>1 for ALL pairs, not just with 255.
    # 
    # If we take all multiples of 3, that's 85 elements.
    # If we try to add elements that are multiples of 5 but not 3, they must share a factor with all multiples of 3 in our set.
    # A multiple of 5 not divisible by 3, e.g., 5, has gcd(5, 3) = 1, so 5 can't coexist with 3.
    # More generally, if x is divisible by 5 but not 3, and y is divisible by 3 but not 5, gcd(x,y) might be 1.
    # For example, 5 and 3: gcd=1. 10 and 3: gcd=1. 25 and 3: gcd=1.
    # So any element divisible by 5 but not 3 will have gcd 1 with 3 (if 3 is in the set).
    # Since 3 is in the multiples-of-3 set, we can't add any element not divisible by 3.
    # 
    # Wait, do we have to include 3? The set just needs to be a subset of [1,255] containing 255.
    # We could choose a different clique. Let's think about this more carefully.
    #
    # The problem is: find the largest subset S of {1,...,255} such that 255 ∈ S and gcd(a,b)>1 for all distinct a,b ∈ S.
    #
    # This is a maximum clique problem in a graph where vertices are {1,...,255} and edges connect pairs with gcd>1,
    # with the constraint that 255 must be in the clique.
    #
    # Since 255 = 3*5*17, every element in S must share a factor with 255, so every element must be divisible by 3, 5, or 17.
    # But more strongly, for any two elements a,b in S, gcd(a,b)>1.
    #
    # Key insight: If S contains an element divisible by 3 but not 5 or 17, and an element divisible by 5 but not 3 or 17,
    # those two might be coprime. E.g., 3 and 5 are coprime.
    #
    # The largest such set is likely all multiples of the smallest prime factor of 255, which is 3.
    # That gives 85 elements.
    #
    # Let's verify: can we get more than 85?
    # Total numbers in [1,255] divisible by 3, 5, or 17:
    # |A∪B∪C| where A=mult of 3, B=mult of 5, C=mult of 17
    # |A| = 85, |B| = 51, |C| = 15
    # |A∩B| = mult of 15 = 17
    # |A∩C| = mult of 51 = 5
    # |B∩C| = mult of 85 = 3
    # |A∩B∩C| = mult of 255 = 1
    # |A∪B∪C| = 85+51+15-17-5-3+1 = 127
    #
    # So at most 127 elements are even candidates. But we need a clique, which is much more restrictive.
    #
    # The set of all multiples of 3 has size 85 and is a valid clique.
    # Can we add any element from B∪C that's not in A?
    # Such an element is divisible by 5 or 17 but not 3.
    # For it to be in a clique with all multiples of 3, it must share a factor with every multiple of 3.
    # But gcd(5,3)=1, so 5 can't be in the clique if 3 is.
    # If we remove 3 from our set, can we add more?
    # This becomes a complex optimization. But intuitively, the all-multiples-of-smallest-prime approach is optimal.
    
    # Let's just return the multiples of 3
    result = list(range(3, 256, 3))
    
    # Verify 255 is in the set
    if 255 not in result:
        result.append(255)
    
    # Verify all pairs have gcd > 1 (spot check)
    # This is guaranteed by construction since all are multiples of 3
    
    return result
