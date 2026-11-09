import time

_T0 = time.time()
_DEADLINE = _T0 + 280.0

def priority(vector, n):
    """
    Priority for a vector in {0,1,2}^n for greedy cap-set construction.
    
    Strategy: Trinary-Balance Routing with High-Quality Hashing.
    
    1. Structural Priority (Layering):
       We prioritize vectors based on how "balanced" their trinary composition is.
       A vector is 'balanced' if the counts of 0s, 1s, and 2s are as close to
       n/3 as possible. This routes the greedy search through the most
       "random" looking parts of the cube, which are less likely to contain
       short linear structures (arithmetic progressions) compared to
       subspaces defined by fixing coordinates to 0 or 2.
       
    2. Intra-Layer Ordering (Refinement):
       Within each balance layer (vectors with the same trinary deviation),
       we use a high-quality deterministic pseudo-random ordering derived
       from a MurmurHash3-style mixing function.
       
    The final priority is a float: layer_score + hash_fraction.
    """
    if time.time() > _DEADLINE:
        return 0.0

    # 1. Calculate structural trinary balance score
    # Count the number of 0s, 1s, and 2s in the vector
    count_0 = 0
    count_1 = 0
    count_2 = 0
    for v in vector:
        if v == 0:
            count_0 += 1
        elif v == 1:
            count_1 += 1
        else:
            count_2 += 1
    
    # Ideal count for each value is n/3.
    # Deviation is the sum of absolute differences from n/3.
    # To avoid floating point, we scale by 3.
    # Deviation_scaled = |count_0*3 - n| + |count_1*3 - n| + |count_2*3 - n|
    # This is an integer.
    dev = abs(count_0 * 3 - n) + abs(count_1 * 3 - n) + abs(count_2 * 3 - n)
    
    # Higher score is better. We want low deviation.
    # Max possible dev is when one count is n, others 0: |3n-n| + |0-n| + |0-n| = 2n + n + n = 4n.
    # Min dev is 0 (if n divisible by 3) or small.
    # So layer_score ranges from n - 4n = -3n to n.
    # This is a valid integer range for layering.
    layer_score = n - dev
    
    # 2. Calculate high-quality hash for tie-breaking
    # Convert vector to a single integer index
    idx = 0
    for v in reversed(vector):
        idx = idx * 3 + v

    # Mix idx and n using MurmurHash3 finalizer logic
    def mix64(h):
        h ^= h >> 33
        h = (h * 0xff51afd7ed558ccd) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 33
        h = (h * 0xc4ceb9fe1a85ec53) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 33
        return h

    # Combine idx and n to decorrelate orders for different dimensions
    seed_val = idx ^ (n * 0x9E3779B97F4A7C15)
    mixed = mix64(seed_val)
    
    # Normalize hash to [0, 1)
    hash_fraction = mixed / (2**64)
    
    # 3. Combine
    # layer_score is an integer. hash_fraction is in [0, 1).
    return float(layer_score + hash_fraction)
