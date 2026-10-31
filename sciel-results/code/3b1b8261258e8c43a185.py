import time
import zlib

# EVOLVE-BLOCK-START

# Global state for deadline tracking
_start_time = None
_DEADLINE = 300.0
_SAFETY_MARGIN = 15.0

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy:
    1. Deadline guard: If time is running out, fall back to a fast hash.
    2. Primary signal: Balance of symbol counts (0, 1, 2). Vectors with counts
       close to n/3 each are more "generic" and less likely to form lines with
       other generic vectors. We use negative sum of squared deviations.
    3. Secondary signal: Distinct symbol count. Vectors using all 3 symbols
       are preferred over those using only 1 or 2, as they are less prone
       to creating collinear triples with common patterns.
    4. Tertiary signal: A fast, well-distributed hash (CRC32) for tie-breaking
       and to avoid structural local optima. This ensures that among vectors
       with similar balance and distinctness, we explore diverse configurations.
    """
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline check: if we're close to the limit, return a fast fallback
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Fast fallback: just use CRC32 hash
        try:
            return float(zlib.crc32(bytes(vector)))
        except Exception:
            return float(sum(vector))
    
    # Count symbols 0, 1, 2
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
    
    # Primary: Balance score (negative squared deviation from n/3)
    ideal = n / 3.0
    sq_dev = (c0 - ideal)**2 + (c1 - ideal)**2 + (c2 - ideal)**2
    # Scale balance to dominate the score
    balance_score = -sq_dev * 1000.0
    
    # Secondary: Distinct symbol count
    distinct_count = 0
    if c0 > 0: distinct_count += 1
    if c1 > 0: distinct_count += 1
    if c2 > 0: distinct_count += 1
    # Scale distinct count to be significant but less than balance
    distinct_score = distinct_count * 100.0
    
    # Tertiary: Fast hash for tie-breaking and diversity
    try:
        hash_val = zlib.crc32(bytes(vector))
    except Exception:
        hash_val = sum(vector) * 31 + n
    # Normalize hash to [0, 1) range for small contribution
    hash_score = (hash_val % 10000) / 10000.0
    
    # Combine scores
    return float(balance_score + distinct_score + hash_score)

# EVOLVE-BLOCK-END
