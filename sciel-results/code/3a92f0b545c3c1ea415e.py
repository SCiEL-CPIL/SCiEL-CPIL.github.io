import time
import zlib

_start_time = None
_DEADLINE = 300 * 0.92  # 276s safety margin before the 300s kill

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.

    Strategy: Smooth Surrogate Heuristic with improved hash tie-breaking.
    1. Primary: Smooth quadratic distinct-symbol score.
    2. Secondary: Smooth variance-based danger proxy.
    3. Tertiary: Quadratic balance penalty.
    4. Quaternary: Symmetry breaking via count of 2s.
    5. Quinary: 64-bit CRC32-based hash for deterministic tie-breaking.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    # Fast symbol counting in one pass
    c0 = 0
    c1 = 0
    c2 = 0
    s = 0
    s2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
            s += 1
            s2 += 1
        else:  # v == 2
            c2 += 1
            s += 2
            s2 += 4

    # Factor 1: Smooth quadratic distinct-symbol score
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Smooth variance-based danger proxy
    mean = s / n
    variance = s2 / n - mean * mean
    danger_score = -variance * 5000.0

    # Factor 3: Quadratic balance penalty
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # Factor 4: Symmetry breaking
    weight_score = c2 * 100.0

    # Factor 5: Fast 64-bit hash for deterministic tie-breaking
    # Use CRC32 twice with different seeds to get a 64-bit hash
    h1 = zlib.crc32(bytes(vector)) & 0xFFFFFFFF
    h2 = zlib.crc32(bytes(vector), 0xAAAAAAAA) & 0xFFFFFFFF
    hash64 = (h1 << 32) | h2
    hash_score = hash64 / 0x10000000000000000

    # Combined priority
    total = distinct_score + weight_score + balance_score + danger_score + hash_score
    return float(total)
