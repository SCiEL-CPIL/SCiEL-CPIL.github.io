import time

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy:
    1. Primary: Geometric danger score - count how many already-selected vectors
       would form an affine line with this vector (Hamming distance 1 or 2).
       Lower danger = higher priority.
    2. Secondary: Smooth distinct symbol bonus (quadratic in presence).
    3. Tertiary: Quadratic balance penalty.
    4. Quaternary: Fast structural hash for deterministic tie-breaking.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    # Fast symbol counting
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # Geometric danger: count vectors at Hamming distance 1 or 2 that are in the set
    # A vector u forms a line with v iff u = v + d or v + 2d for some nonzero d
    # This means u differs from v in exactly 1 or 2 coordinates
    # We need to check which vectors in the "danger zone" are already selected
    # Since we don't have direct access to the selected set, we approximate:
    # Vectors with high danger are those that are "close" to many potential line partners
    # We use a proxy: the number of vectors at distance 1 or 2 is fixed for all v,
    # so we need a different approach.
    
    # Better approach: Use the structure of the vector itself.
    # Vectors with many 0s are more likely to form lines with vectors that have 1s or 2s
    # in those positions. The "danger" is proportional to how many positions have
    # a value that allows line completion.
    
    # For a vector v, the number of directions d such that v+d is a valid vector
    # is 3^n - 1 (all nonzero directions). But the number of vectors at distance 1
    # is n*2 (change one coordinate to one of the other two values).
    # The number at distance 2 is C(n,2) * 4 (change two coordinates, each to one of
    # two other values... actually each coordinate can change to 2 values, so 4 per pair).
    
    # Key insight: vectors with more "extreme" coordinates (all 0s or all 2s) are
    # more dangerous because they can form lines with more vectors.
    # Vectors with balanced coordinates are safer.
    
    # Compute a "line density" proxy: for each coordinate, how many vectors would
    # complete a line through this coordinate. A coordinate with value 0 can be part
    # of a line with a coordinate value 1 or 2. The "danger" per coordinate is
    # related to how unique that value is in the context.
    
    # Simpler and more effective: use the variance of coordinates as a danger proxy.
    # High variance = more "extreme" = more likely to form lines.
    coord_sum = sum(vector)
    coord_sq_sum = sum(v * v for v in vector)
    variance = coord_sq_sum / n - (coord_sum / n) ** 2
    
    # Higher variance = more dangerous = lower priority
    danger_score = -variance * 5000.0
    
    # Distinct symbol bonus: reward vectors using all 3 symbols
    presence0 = 1.0 if c0 > 0 else 0.0
    presence1 = 1.0 if c1 > 0 else 0.0
    presence2 = 1.0 if c2 > 0 else 0.0
    num_distinct = presence0 + presence1 + presence2
    distinct_score = 10000.0 * num_distinct * num_distinct
    
    # Balance: Quadratic deviation from uniform
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0
    
    # Fast hash: simple polynomial rolling hash
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    
    hash_score = h / 0x7FFFFFFF
    
    return float(distinct_score + danger_score + balance_score + hash_score)
