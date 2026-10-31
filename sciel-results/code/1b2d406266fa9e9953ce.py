# EVOLVE-BLOCK-START
def priority(vector, n):
    # Structured priority to guide greedy cap-set construction.
    # Primary tier: coordinate-sum mod 3 (group compatible vectors).
    # Secondary: prefer vectors with more 1s (sparsely conflicting),
    # then fewer 2s, with a deterministic tie-break for stability.
    s = 0
    n1 = 0
    n2 = 0
    for x in vector:
        s += x
        if x == 1:
            n1 += 1
        elif x == 2:
            n2 += 1
    tier = s % 3
    # Higher tier value -> higher priority (we want a specific order).
    # We order tiers so that sum%3 == 0 first, then 1, then 2.
    tier_score = (3 - tier) * 10000  # tier0->30000, tier1->20000, tier2->10000
    # Within tier, prefer more 1s, then fewer 2s.
    shape_score = n1 * 100 - n2
    # Deterministic small tie-break from lexicographic-ish hash.
    tie = 0
    for i, x in enumerate(vector):
        tie = (tie * 3 + x) % 997
    return tier_score + shape_score + tie * 1e-4
# EVOLVE-BLOCK-END
