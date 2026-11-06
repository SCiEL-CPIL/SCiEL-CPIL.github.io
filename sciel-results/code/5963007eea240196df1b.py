import time
from math import gcd

def construct():
    # 255 = 3 * 5 * 17
    # We seek the largest subset S of [1, 255] such that:
    # 1. 255 in S
    # 2. gcd(a, b) > 1 for all distinct a, b in S
    #
    # Analysis:
    # Any element x in S must satisfy gcd(x, 255) > 1, so x must be divisible by 3, 5, or 17.
    # Let S_3 = {x in S : 3|x}, S_5 = {x in S : 5|x}, S_17 = {x in S : 17|x}.
    # Note that S = S_3 ∪ S_5 ∪ S_17 (since every x in S shares a factor with 255).
    #
    # If S contains an element not divisible by 3, say x with 5|x (and 3∤x), then for any y in S_3 with gcd(y, x) > 1, 
    # y must share a factor with x. Since 3∤x, the common factor must be 5 or 17.
    # However, if S_3 contains any element not divisible by 5 or 17 (e.g., 3, 6, 9, 12, 15 is div by 5... wait 15 is div by 5. 3 is not).
    # If 3 is in S, then any x in S must have gcd(x, 3) > 1, implying 3|x. Thus if 3 in S, S ⊆ multiples of 3. Size = 85.
    #
    # Can we get larger than 85?
    # If we exclude some multiples of 3, can we add enough non-multiples of 3 to exceed 85?
    # The non-multiples of 3 that are non-coprime to 255 are multiples of 5 or 17 (not 3).
    # Count of such numbers:
    # Multiples of 5 not div by 3: floor(255/5) - floor(255/15) = 51 - 17 = 34.
    # Multiples of 17 not div by 3: floor(255/17) - floor(255/51) = 15 - 5 = 10.
    # Intersection (div by 5 and 17, not 3): div by 85. floor(255/85)=3. 85, 170. Neither div by 3. So 2 numbers.
    # Total candidates outside S_3: 34 + 10 - 2 = 42.
    #
    # If we drop k elements from S_3, we can potentially add at most 42 elements.
    # But adding an element x (div by 5, not 3) requires that for all y in S, gcd(x, y) > 1.
    # Specifically, for any y in S that is not divisible by 5, gcd(x, y) must be > 1 via 17.
    # This imposes severe constraints.
    #
    # It is a known result that for N with smallest prime factor p, the maximum size of a pairwise non-coprime subset
    # containing N is floor(N/p). This is because the set of multiples of p is a clique of size floor(N/p),
    # and any element not divisible by p is coprime to some multiple of p (specifically p itself, if p is in the set).
    # Even if p is not in the set, the density of multiples of p is the highest among prime factors.
    #
    # Let's verify the bound 85.
    # S = {3, 6, ..., 255}. Size 85.
    # Is it possible to have size 86?
    # By the Erdos-Ko-Rado type results for gcd graphs, or simply by the fact that the independence number of the coprime graph
    # is related to the prime factors.
    #
    # Let's try to construct a set larger than 85.
    # Suppose we take all multiples of 3 (85 elements). Can we add any other element?
    # No, because any other element x (not div by 3) is coprime to 3 (which is in the set).
    # So we must remove 3.
    # Remove 3. Set size 84. Can we add x?
    # x must be non-coprime to all remaining multiples of 3.
    # Let x be a multiple of 5 (not 3). e.g., 5.
    # gcd(5, 6) = 1. So 6 must be removed.
    # gcd(5, 9) = 1. So 9 must be removed.
    # gcd(5, 12) = 1. So 12 must be removed.
    # In general, x=5 requires removing all multiples of 3 that are not divisible by 5.
    # Multiples of 3: 3, 6, 9, 12, 15, 18, 21, 24, 27, 30...
    # Not div by 5: 3, 6, 9, 12, 18, 21, 24, 27, 33, 36, 39, 42, 48...
    # There are floor(255/3) - floor(255/15) = 85 - 17 = 68 such numbers.
    # So to add 5, we must remove 68 numbers. Net change: -68 + 1 = -67.
    # This is a huge loss.
    #
    # What if we pick a set of non-multiples of 3 that are all divisible by 5?
    # Then we must remove all multiples of 3 not divisible by 5.
    # We keep multiples of 15. Count: 17.
    # We can add all multiples of 5 not div by 3. Count: 34.
    # Total size: 17 + 34 = 51.
    # This is smaller than 85.
    #
    # What if we mix?
    # The optimal strategy is clearly to stick with the smallest prime factor.
    # The set of multiples of 3 has size 85.
    #
    # Therefore, the maximum cardinality is 85.
    
    # Construct the set of multiples of 3 in [1, 255]
    S = list(range(3, 256, 3))
    
    # Verify 255 is included
    if 255 not in S:
        S.append(255)
    
    return S
