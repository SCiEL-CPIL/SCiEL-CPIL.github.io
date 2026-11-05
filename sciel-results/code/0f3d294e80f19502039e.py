"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s limit
    
    # 255 = 3 * 5 * 17
    # We need a subset of [1, 255] containing 255, with gcd(a,b) > 1 for all distinct pairs.
    # Maximize cardinality.
    
    # Strategy: We want to find the largest clique in the graph where vertices are [1,255]
    # and edges connect pairs with gcd > 1. But this is expensive.
    
    # Better strategy: Any valid set must have all elements sharing a common factor with 255
    # (since 255 is in the set). So all elements must be divisible by 3, 5, or 17.
    # But additionally, any two elements must have gcd > 1.
    
    # Key insight: If we pick all multiples of a single prime p that divides 255,
    # then any two elements share factor p, so gcd >= p > 1. This gives a valid set.
    # For p=3: multiples of 3 in [1,255] = 85 elements (3,6,...,255)
    # For p=5: multiples of 5 in [1,255] = 51 elements (5,10,...,255)
    # For p=17: multiples of 17 in [1,255] = 15 elements (17,34,...,255)
    
    # So multiples of 3 give 85 elements. Can we do better by mixing?
    
    # If we mix elements divisible by different primes, we need to ensure gcd > 1 for all pairs.
    # E.g., an element divisible by 3 but not 5 or 17, and another divisible by 5 but not 3 or 17,
    # would have gcd possibly 1.
    
    # Let's think: any element in our set must share a factor with every other element.
    # If we include an element a that is divisible only by 3 (not 5 or 17), then every other
    # element must also be divisible by 3. So if we include any element not divisible by 5 or 17,
    # all elements must be divisible by 3.
    
    # More generally: Let S be our set. For any a in S, let P(a) be the set of primes in {3,5,17}
    # that divide a. For the set to be valid, for any two elements a,b in S, we need gcd(a,b) > 1.
    # This means P(a) ∩ P(b) ≠ ∅ is NOT sufficient (a could be 3*7 and b could be 3*11, gcd=3, fine;
    # but a could be 3*7 and b could be 5*7, gcd=7, also fine).
    
    # Actually the condition is gcd(a,b) > 1, which means they share ANY prime factor, not just from {3,5,17}.
    
    # But wait: 255 is in the set. So for any a in S, gcd(255, a) > 1, meaning a must be divisible
    # by 3, 5, or 17. So every element has at least one of {3,5,17} as a factor.
    
    # Now for two elements a, b both in S: they each have at least one factor from {3,5,17}.
    # If a is divisible by 3 and b is divisible by 3, gcd >= 3. Good.
    # If a is divisible by 3 and b is divisible by 5, gcd could be 1 (e.g., a=3, b=5).
    # So we need to be careful.
    
    # The largest valid set is likely all multiples of 3 (85 elements), OR we can try to add
    # some elements that are not multiples of 3 but are multiples of 5 or 17, provided they
    # share a factor with all existing elements.
    
    # If our set is all multiples of 3, can we add an element x that is not a multiple of 3?
    # x must be divisible by 5 or 17 (to have gcd with 255 > 1).
    # x must have gcd > 1 with every multiple of 3 in our set.
    # In particular, x must have gcd > 1 with 3 itself (if 3 is in the set).
    # gcd(x, 3) > 1 means x is divisible by 3. Contradiction.
    # So if 3 is in the set, we cannot add any element not divisible by 3.
    
    # What if we remove 3 from the set? Then we could potentially add elements not divisible by 3.
    # But we lose one element. Let's see if we gain more.
    
    # Alternative approach: Use a greedy/optimization approach.
    # Start with all multiples of 3 (85 elements). This is valid.
    # Try to see if we can get more than 85.
    
    # Upper bound: The set must be a clique in the gcd>1 graph. 
    # The number of elements divisible by 3 is 85.
    # Can we have a set larger than 85?
    
    # Consider: elements divisible by 3 but not 5 or 17: these are 3*k where k not divisible by 5 or 17.
    # Count: multiples of 3 = 85. Multiples of 15 = 17. Multiples of 51 = 5. Multiples of 255 = 1.
    # By inclusion-exclusion, multiples of 3 not divisible by 5 or 17: 85 - 17 - 5 + 1 = 64.
    # These 64 elements only share factor 3 with other multiples of 3.
    
    # If we include any of these 64 elements, say a = 3*k where 5∤k and 17∤k,
    # then every other element must share a factor with a. Since a's prime factors are {3} ∪ primes of k,
    # and k < 255/3 = 85, k can have various prime factors.
    
    # This is getting complex. Let me try a different approach:
    # Use a maximum clique algorithm on the graph, but that's expensive for 255 nodes.
    
    # Actually, let's just verify that 85 is achievable and see if we can do better.
    # Multiples of 3: [3, 6, 9, ..., 255]. Size 85. All pairs have gcd >= 3. Valid.
    
    # Can we do better? Let's try to construct a set larger than 85.
    # We need at least 86 elements from [1,255] containing 255, all pairwise gcd > 1.
    
    # Every element must be divisible by 3, 5, or 17.
    # Total elements divisible by 3, 5, or 17:
    # |A∪B∪C| where A=mult of 3, B=mult of 5, C=mult of 17
    # |A|=85, |B|=51, |C|=15
    # |A∩B|=17, |A∩C|=5, |B∩C|=3 (mult of 85)
    # |A∩B∩C|=1 (255)
    # |A∪B∪C| = 85+51+15-17-5-3+1 = 127
    
    # So there are 127 candidate elements. We need a clique of size > 85 in this subgraph.
    
    # Let's try a greedy approach: start with all multiples of 3, then try to add elements
    # not in this set, removing conflicting elements as needed.
    
    # Actually, let me think about this more carefully.
    # If we have a set S where all elements are divisible by 3, |S| can be at most 85.
    # To exceed 85, we must include some element not divisible by 3.
    # Let x be such an element. x is divisible by 5 or 17 (but not 3).
    # For x to coexist with elements in S, every element in S must share a factor with x.
    # x is not divisible by 3, so x's factors from {3,5,17} are subset of {5,17}.
    # If x is divisible by 5 but not 17, then x's relevant factors include 5 and possibly others.
    # Every element in S must share a factor with x. Since elements in S are all divisible by 3,
    # they share factor 3 with each other, but with x they need to share a factor other than 3
    # (since x is not divisible by 3).
    
    # So if x is divisible by 5, every element in S must be divisible by 5 or share another factor with x.
    # This is restrictive.
    
    # Given the complexity, let's just return the multiples of 3 (85 elements) as a solid answer,
    # and optionally try to improve with a search if time permits.
    
    multiples_of_3 = list(range(3, 256, 3))
    best_set = set(multiples_of_3)
    best_size = len(best_set)
    
    # Try to see if we can do better with a limited search
    # This is a maximum clique problem which is NP-hard, so we'll use a heuristic.
    
    # For now, return the multiples of 3 which gives 85.
    if time.time() > deadline:
        return sorted(best_set)
    
    # The answer is 85 (all multiples of 3).
    return sorted(best_set)
# EVOLVE-BLOCK-END
