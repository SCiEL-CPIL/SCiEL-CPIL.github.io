"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

def construct():
    """
    Return a nonempty subset of [1, 255] containing 255,
    with gcd(a, b) > 1 for all distinct pairs. Maximize cardinality.
    """
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    # 255 = 3 * 5 * 17
    # Any element must share a prime factor with 255 (else gcd with 255 = 1)
    # So elements must be divisible by 3, 5, or 17.
    # For pairwise gcd > 1, all pairs must share a common prime factor.
    # The optimal strategy is to pick all multiples of the smallest prime factor of 255, which is 3.
    # This gives floor(255/3) = 85 elements, all sharing factor 3.
    
    # Verify: 255 is in the set? 255 % 3 == 0 → yes.
    # Any two multiples of 3 have gcd >= 3 > 1 → valid.
    
    # Can we do better than 85?
    # The set must be a clique in the graph where edge (a,b) exists iff gcd(a,b)>1.
    # All elements must be from {n in [1,255] : gcd(n,255)>1}.
    # Within that set, we need a maximal clique.
    # The multiples of 3 form a clique of size 85.
    # Could a larger clique exist mixing multiples of 5 or 17?
    # Any number divisible by 5 but not 3 has gcd 1 with any multiple of 3 that isn't divisible by 5.
    # So we can't freely mix. The largest single-prime clique is multiples of 3: 85.
    # This is optimal.
    
    result = list(range(3, 256, 3))
    
    # Safety check
    if len(result) == 0 or 255 not in result:
        return [255]
    
    return result

# EVOLVE-BLOCK-END
