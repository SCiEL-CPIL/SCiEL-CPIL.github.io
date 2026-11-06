"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # We want a nonempty subset S of [1,100] with no subset summing to 200.
    # Maximize |S|.
    #
    # Key insight: The set {66, 67, ..., 100} has 35 elements.
    # The minimum sum of any 3 elements is 66+67+68 = 201 > 200.
    # The maximum sum of any 2 elements is 100+99 = 199 < 200.
    # No single element equals 200 (max is 100).
    # Therefore, no subset of {66..100} sums to exactly 200.
    #
    # Can we do better? We need to remove at least 65 elements from [1,100].
    # The construction {66..100} removes exactly 65 elements (1..65).
    #
    # Let's verify this is optimal. If we have a set of size 36, we removed only 64 elements.
    # By the pigeonhole principle / subset sum arguments, it's hard to avoid sum 200
    # with 36 elements from [1,100].
    #
    # Actually, let's just use the deterministic construction {66..100} which is proven valid.
    # It's fast, reliable, and likely optimal.
    
    # Construct the set {66, 67, ..., 100}
    S = list(range(66, 101))
    
    # Verify: no subset sums to 200
    def verify(S):
        S = list(S)
        reachable = {0}
        for x in S:
            if x > 200:
                continue
            new = {s + x for s in reachable if s + x <= 200}
            reachable |= new
        return 200 not in reachable
    
    if not verify(S):
        # Fallback (should never happen)
        S = [1]
    
    return S
# EVOLVE-BLOCK-END
