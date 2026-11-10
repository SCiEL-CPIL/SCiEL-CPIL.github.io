"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
from itertools import combinations

def _is_dissociated_reciprocal(subset):
    """
    Check if all subset sums of reciprocals of elements in 'subset' are distinct.
    A set S is dissociated (with respect to reciprocal sums) if the map
    {0,1}^n -> Q, x -> sum(x_i / s_i) is injective.
    
    Equivalently, no two distinct subsets have the same sum of reciprocals.
    We check this by verifying that all 2^|S| subset sums are distinct.
    For efficiency, we can use the fact that if the set is dissociated,
    then the vectors (1/s_i) must be such that no non-trivial signed combination
    sums to zero. But the most direct check for small sets is to enumerate all
    subset sums.
    
    For larger sets, we use a linear independence check over Q:
    The set {1/s_1, ..., 1/s_n} is linearly independent over Z (in the sense
    that no subset sum collision exists) iff the only integer solution to
    sum(c_i / s_i) = 0 with c_i in {-1, 0, 1} is c_i = 0 for all i.
    
    However, the standard definition of a dissociated set requires that all
    subset sums are distinct, which is equivalent to: for any two distinct
    subsets A, B of S, sum_{a in A} 1/a != sum_{b in B} 1/b.
    This is equivalent to: for any non-empty subset T of S and any assignment
    of signs, sum_{t in T} sign_t / t != 0 unless T is empty.
    
    A more efficient check: the 2^n subset sums are all distinct iff the n
    elements 1/s_i are linearly independent over the field Q in a specific sense.
    Actually, for real numbers, the subset sums are all distinct iff the elements
    are linearly independent over Q with coefficients in {-1, 0, 1}. But a simpler
    sufficient and necessary condition for distinct subset sums is that the n
    vectors are linearly independent over Q when considered as elements of Q^n? No.
    
    Let me think again. We have n real numbers r_1, ..., r_n. The 2^n subset sums
    are distinct iff there is no non-trivial relation sum_{i in T} eps_i r_i = 0
    where eps_i in {-1, 1} and T is non-empty. This is equivalent to the r_i being
    linearly independent over Q with the constraint that coefficients are in {-1, 0, 1}.
    
    For our purpose, since we're working with small n (up to 22), we can use a
    greedy approach: try to add elements one by one, and for each candidate, check
    if adding it preserves the dissociated property. The check can be done by
    verifying that the new element's reciprocal is not equal to the difference of
    any two existing subset sums... but that's expensive.
    
    Alternative: use the fact that for a set to have all distinct subset sums of
    reciprocals, it suffices to check that the reciprocal of each new element is
    not in the set of differences of existing subset sums. But this is still complex.
    
    For n <= 22, a practical approach: maintain the set of all subset sums as a set
    of Fractions. When adding a new element with reciprocal r, the new subset sums
    are old_sums union {s + r for s in old_sums}. The new set is dissociated iff
    |new_sums| == 2 * |old_sums|, i.e., no collisions.
    
    This is O(2^n) per check, which is feasible for n up to ~20-22 if we're careful.
    """
    n = len(subset)
    if n == 0:
        return True
    
    # Start with the empty sum
    sums = {Fraction(0, 1)}
    
    for s in subset:
        r = Fraction(1, s)
        new_sums = set()
        for existing in sums:
            new_sums.add(existing)
            new_sums.add(existing + r)
        
        if len(new_sums) != 2 * len(sums):
            return False
        sums = new_sums
    
    return True

def construct():
    """
    Find a nonempty subset of [1, 22] with maximum cardinality such that
    all reciprocal subset sums (including empty sum = 0) are distinct.
    
    Use greedy: try to build the largest possible set by adding elements
    in a strategic order. We try multiple orderings and pick the best.
    """
    candidates = list(range(1, 23))  # [1, 2, ..., 22]
    
    best_set = []
    
    # Try different ordering strategies
    # Strategy 1: descending order (larger elements first, smaller reciprocals)
    # Strategy 2: ascending order
    # Strategy 3: random-ish orders via different seeds
    
    import random
    
    orderings = []
    # Descending
    orderings.append(sorted(candidates, reverse=True))
    # Ascending
    orderings.append(sorted(candidates))
    # Some random orderings
    rng = random.Random(42)
    for _ in range(20):
        order = candidates[:]
        rng.shuffle(order)
        orderings.append(order)
    
    for order in orderings:
        current = []
        current_sums = {Fraction(0, 1)}
        
        for candidate in order:
            r = Fraction(1, candidate)
            # Check if adding candidate preserves distinctness
            new_sums = set()
            for existing in current_sums:
                new_sums.add(existing)
                new_sums.add(existing + r)
            
            if len(new_sums) == 2 * len(current_sums):
                # No collision, add it
                current.append(candidate)
                current_sums = new_sums
        
        if len(current) > len(best_set):
            best_set = current[:]
    
    # If best_set is empty (shouldn't happen), return at least one element
    if not best_set:
        best_set = [1]
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
