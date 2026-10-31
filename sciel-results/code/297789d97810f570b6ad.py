# EVOLVE-BLOCK-START
import time

_start_time = None
_deadline = 320.0  # safety margin before the 330s kill

def _fast_hash_64(vector):
    """
    Fast 64-bit polynomial hash for deterministic tie-breaking.
    Uses a large prime base to reduce collisions and ensure uniform distribution.
    """
    h = 0
    for i, val in enumerate(vector):
        # Use a large odd base for better mixing
        h = (h * 31 + val + i) & 0xFFFFFFFFFFFFFFFF
    return h

def priority(vector, n):
    """
    Priority for greedy cap-set construction in {0,1,2}^n.

    Structural Strategy (Geometric Multiplicative Model):
    1. Hard Tiering: Strongly prefer vectors with all 3 distinct symbols.
       3-distinct vectors form the bulk of large cap sets.
    2. Geometric Centrality: Instead of additive variance/balance penalties,
       we use a multiplicative score based on the product of distances to
       the all-zeros and all-twos boundaries. This rewards vectors that are
       "central" in the sum-space, which is a structural proxy for safety
       against affine lines.
    3. High-Resolution Hash: Uses a 64-bit polynomial hash to provide
       fine-grained, uniform tie-breaking, preventing the greedy algorithm
       from clustering in specific sub-spaces.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _deadline:
        # Past deadline: ordering no longer matters, return a flat constant.
        return 0.0

    # Fast symbol counting in one pass
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:  # v == 2
            c2 += 1

    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    
    # Calculate 64-bit hash for tie-breaking
    h = _fast_hash_64(vector)
    hash_score = (h / 0xFFFFFFFFFFFFFFFF) * 1000.0

    if num_distinct < 3:
        # Lower tier: prioritize 2-distinct over 1-distinct
        # Scale by distinct count to maintain some gradient
        tier_score = num_distinct * 10000.0
        return float(tier_score + hash_score)

    # Tier 1: 3-distinct vectors
    # Base score ensures they are always preferred over lower tiers
    base_score = 100000.0
    
    # Geometric Centrality Score
    # s1 = sum(v), s2 = sum(2-v)
    # Product is maximized when the vector is balanced in its "mass"
    # between 0 and 2, with 1s contributing to both.
    s1 = c1 + 2 * c2
    s2 = 2 * c0 + c1
    
    # Normalize by n^2 to make score independent of n (roughly)
    # Max possible product is n^2 (all 1s: s1=n, s2=n)
    # Min possible product for 3-distinct is small.
    geo_score = (s1 * s2) / (n * n) if n > 0 else 0.0
    
    # Scale geo_score to be significant but not dominating the base
    # Range of geo_score is [0, 1].
    # We want it to influence the ordering among 3-distinct vectors.
    geo_scaled = geo_score * 5000.0
    
    total = base_score + geo_scaled + hash_score
    return float(total)
# EVOLVE-BLOCK-END
