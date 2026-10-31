"""Cap set priority: target central, diverse vectors to avoid boundary APs."""
import time
import math

_start_time = None
_DEADLINE_FRACTION = 0.95  # 5% safety margin


def _soft_min(a, b, k=1.0):
    """Smooth approximation of min(a, b) using log-sum-exp trick."""
    m = min(a, b)
    if a == b:
        return a
    # soft-min = -soft_max(-a, -b)
    # soft_max(-a, -b) = log(exp(-a) + exp(-b))
    # For numerical stability:
    ea = math.exp(-(a - m))
    eb = math.exp(-(b - m))
    return m - math.log(ea + eb) / k


def priority(vector, n):
    """
    Priority based on product of distances to all-zeros and all-twos vectors,
    plus a smooth diversity bonus for having all three coordinate values present.
    
    Vectors with many 1s and balanced 0s/2s are prioritized because they're
    geometrically central and less likely to form 3-term APs with boundary vectors.
    Uses soft-min to create a smooth priority landscape.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're close to the limit, return a safe fallback
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
    
    # Product of distances to all-zeros and all-twos
    dist_to_zero = count_1 + 2 * count_2
    dist_to_two = 2 * count_0 + count_1
    center_score = dist_to_zero * dist_to_two
    
    # Smooth diversity bonus using soft-min
    # Soft-min of the three counts gives a smooth measure of balance
    sm01 = _soft_min(count_0, count_1, k=2.0)
    sm012 = _soft_min(sm01, count_2, k=2.0)
    
    # Smooth indicator for "all three present" using soft-max of indicators
    # Instead of hard if/else, use smooth transitions
    # Indicator that at least two types are present (smooth)
    pair_min_01 = _soft_min(count_0, count_1, k=3.0)
    pair_min_02 = _soft_min(count_0, count_2, k=3.0)
    pair_min_12 = _soft_min(count_1, count_2, k=3.0)
    
    # Smooth max of pairwise mins = smooth "at least two types present" measure
    # Use soft-max via log-sum-exp
    m = max(pair_min_01, pair_min_02, pair_min_12)
    if m == 0:
        two_type_measure = 0.0
    else:
        ep01 = math.exp((pair_min_01 - m) / 2.0)
        ep02 = math.exp((pair_min_02 - m) / 2.0)
        ep12 = math.exp((pair_min_12 - m) / 2.0)
        two_type_measure = m + 2.0 * math.log(ep01 + ep02 + ep12)
    
    # Smooth "all three present" indicator
    # Use the soft-min of all three as a smooth proxy
    all_three_measure = sm012
    
    # Blend: full diversity when all three present, partial when only two
    # Weight by how "complete" the set is
    diversity_bonus = 2.0 * all_three_measure + 0.5 * two_type_measure
    
    return center_score + diversity_bonus
# EVOLVE-BLOCK-END
