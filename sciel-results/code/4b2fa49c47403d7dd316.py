"""Cap set priority: Chaos-breaking hybrid with smooth structural bias."""
import time
import hashlib
import math

_start_time = None
_DEADLINE_FRACTION = 0.95  # 5% safety margin


def _soft_min(a, b, k=1.0):
    """Smooth approximation of min(a, b) using log-sum-exp trick."""
    m = min(a, b)
    if a == b:
        return a
    # soft-min = -soft_max(-a, -b)
    # For numerical stability:
    ea = math.exp(-(a - m))
    eb = math.exp(-(b - m))
    return m - math.log(ea + eb) / k


def priority(vector, n):
    """
    Hybrid priority:
    1. Chaos component: High-quality hash (SHA256) to randomize order and break greedy traps.
    2. Structural component: Smooth measure of centrality and balance to avoid boundary APs.
    
    The blend favors chaos (0.85) to ensure diverse exploration, with a structural bias (0.15)
    to prefer 'safe' central vectors.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're close to the limit, return a fast fallback
    if time.time() - _start_time > _DEADLINE_FRACTION * 330:
        # Fallback: simple center score only
        count_1 = 0
        count_2 = 0
        for v in vector:
            if v == 1:
                count_1 += 1
            elif v == 2:
                count_2 += 1
        count_0 = n - count_1 - count_2
        dist_to_zero = count_1 + 2 * count_2
        dist_to_two = 2 * count_0 + count_1
        return float(dist_to_zero * dist_to_two)

    # 1. Chaos Component (Hash)
    # Convert vector to bytes for hashing.
    # Vector elements are in {0, 1, 2}, so they fit in a single byte.
    try:
        v_bytes = bytes(vector)
    except TypeError:
        # Fallback if vector is not a tuple of ints (defensive)
        v_bytes = str(vector).encode('utf-8')
        
    h = hashlib.sha256(v_bytes).digest()
    # Take first 8 bytes to create a 64-bit integer
    hash_val = int.from_bytes(h[:8], 'big')
    # Normalize to [0, 1]
    random_score = hash_val / (2**64)

    # 2. Structural Component (Smooth Safety)
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
            
    # Center score: product of distances to all-zeros and all-twos
    dist_to_zero = count_1 + 2 * count_2
    dist_to_two = 2 * count_0 + count_1
    center_score = dist_to_zero * dist_to_two
    
    # Normalize center score to [0, 1]
    max_center = n * n
    if max_center > 0:
        norm_center = center_score / max_center
    else:
        norm_center = 0.0
        
    # Smooth balance bonus using soft-min
    sm01 = _soft_min(count_0, count_1, k=2.0)
    sm012 = _soft_min(sm01, count_2, k=2.0)
    
    # Normalize balance to [0, 1]
    max_balance = n / 3.0
    if max_balance > 0:
        norm_balance = sm012 / max_balance
    else:
        norm_balance = 0.0
        
    # Combine structural scores
    # Weight center heavily, balance slightly
    structural_score = 0.8 * norm_center + 0.2 * norm_balance
    
    # 3. Final Priority
    # Blend: Mostly random (chaos), slightly structural (safety)
    final_priority = 0.85 * random_score + 0.15 * structural_score
    
    return final_priority
# EVOLVE-BLOCK-END
