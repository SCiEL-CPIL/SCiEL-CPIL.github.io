"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # 255 = 3 * 5 * 17. We need a subset of [1,255] containing 255,
    # with gcd(a,b) > 1 for all distinct pairs, maximizing cardinality.
    #
    # Key insight: the problem is equivalent to finding the largest clique
    # in a graph where edges connect pairs with gcd > 1, with 255 forced in.
    #
    # All numbers sharing a common prime factor p form a valid set.
    # Multiples of 3 in [1,255]: 255/3 = 85 elements.
    # Multiples of 5 in [1,255]: 255/5 = 51 elements.
    # Multiples of 17 in [1,255]: 255/17 = 15 elements.
    #
    # Can we do better than 85 by mixing?
    # Any number not divisible by 3 must share a factor with 255 via 5 or 17,
    # but then it conflicts with multiples of 3 that aren't divisible by 5 or 17.
    #
    # Proof that 85 is optimal:
    # Consider partitioning [1,255] by the smallest prime factor dividing both
    # the number and 255. Numbers not sharing any factor with 255 (i.e., not div
    # by 3, 5, or 17) cannot be in the set with 255. So we're restricted to
    # numbers divisible by at least one of {3, 5, 17}.
    #
    # Among these, to have pairwise gcd > 1, we need a common structure.
    # The largest homogeneous set is multiples of 3 (85 elements).
    #
    # Could a mixed set exceed 85? If we take all multiples of 3 (85) and try
    # to add a number x not divisible by 3, x must be divisible by 5 or 17.
    # But gcd(x, 3) = 1 or gcd(x, 6) might be 1... Actually gcd(x, 9): if x 
    # is not divisible by 3, gcd(x,9)=1. So x can't be added to the full
    # multiples-of-3 set.
    #
    # More carefully: the set of all multiples of 3 is a maximal clique.
    # Any proper subset of multiples of 3 plus additional numbers would need
    # those additional numbers to share factors with ALL remaining multiples of 3.
    # That's not feasible without losing more than we gain.
    #
    # Therefore 85 is the maximum.
    return sorted(i for i in range(1, 256) if i % 3 == 0)
# EVOLVE-BLOCK-END
