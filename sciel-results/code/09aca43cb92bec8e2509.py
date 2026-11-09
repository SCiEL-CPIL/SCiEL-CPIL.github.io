import time

_T0 = time.time()
_DEADLINE = _T0 + 280.0

def priority(vector, n):
    """
    Priority for a vector in {0,1,2}^n for greedy cap-set construction.
    
    Strategy: Refined Count-of-Ones Routing with 0/2 Balance.
    
    1. Primary Structural Priority:
       Prioritize vectors where the count of 1s is close to n/2.
       This was found to be superior to full trinary balance.
       
    2. Secondary Structural Priority:
       Among vectors with similar count of 1s, prioritize those where
       the count of 0s and 2s are balanced (close to each other).
       This avoids biasing towards subspaces like {0,1}^n or {1,2}^n.
       
    3. Intra-Layer Ordering:
       Use a high-quality deterministic pseudo-random ordering derived
       from a MurmurHash3-style mixing function.
    """
    if time.time() > _DEADLINE:
        return 0.0

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
    
    # Primary: Distance of count_1 from n/2.
    # Scaled by 2 to keep integer: |2*count_1 - n|
    primary_dev = abs(2 * count_1 - n)
    
    # Secondary: Imbalance between count_0 and count_2.
    # |count_0 - count_2|
    secondary_dev = abs(count_0 - count_2)
    
    # Combine into a single integer layer score.
    # We want low primary_dev and low secondary_dev.
    # Max primary_dev is n (if count_1=0 or n).
    # Max secondary_dev is n - count_1 (if all non-1s are 0 or 2).
    # Weight primary more heavily.
    # Let's say primary_dev * 10 + secondary_dev.
    layer_penalty = primary_dev * 10 + secondary_dev
    
    # Convert to score (higher is better).
    # Max penalty is roughly 10n + n = 11n.
    # So layer_score = 11n - layer_penalty.
    layer_score = 11 * n - layer_penalty

    # Hash for tie-breaking
    idx = 0
    for v in reversed(vector):
        idx = idx * 3 + v

    def mix64(h):
        h ^= h >> 33
        h = (h * 0xff51afd7ed558ccd) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 33
        h = (h * 0xc4ceb9fe1a85ec53) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 33
        return h

    seed_val = idx ^ (n * 0x9E3779B97F4A7C15)
    mixed = mix64(seed_val)
    
    hash_fraction = mixed / (2**64)
    
    return float(layer_score + hash_fraction)
