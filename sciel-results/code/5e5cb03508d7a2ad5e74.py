# EVOLVE-BLOCK-START
import time
import math

_start_time = None
_deadline = 300.0  # safety margin before the 330s kill

def priority(vector, n):
    """
    Priority for greedy cap-set construction in {0,1,2}^n.
    
    Structural approach: Minimize affine-line participation potential.
    A coordinate position i is "AP-capable" if the vector has a value v at i
    such that there exist other vectors that could complete (v, v+1, v+2) or
    (v, v-1, v-2) patterns. We approximate danger by counting coordinates
    where the value is 0 or 2 (since 1 is the "middle" and safest).
    
    Key structural insight: Vectors with more 1s are safer because 1 cannot
    be an endpoint of a 3-term AP in that coordinate (only middle). So we
    strongly prefer vectors with high count of 1s, then break ties by
    preferring balanced 0/2 distributions to avoid clustering.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _deadline:
        return 0.0

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

    # Primary: maximize count of 1s (safest symbol - can't be AP endpoint)
    # This is the dominant structural factor
    one_score = c1 * 10000.0

    # Secondary: among vectors with same c1, prefer balanced 0/2 split
    # to avoid having too many 0s or too many 2s (which could form APs
    # with vectors that have the complementary pattern)
    if c0 + c2 > 0:
        balance = abs(c0 - c2)
        balance_score = -balance * 100.0
    else:
        balance_score = 0.0

    # Tertiary: deterministic hash for tie-breaking
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    total = one_score + balance_score + hash_score
    return float(total)
# EVOLVE-BLOCK-END
