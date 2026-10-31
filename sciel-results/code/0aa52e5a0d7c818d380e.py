"""Cap set priority: center-distance + smooth surrogates for geometric separation."""
import time

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin


def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.

    Strategy (combined design):
    1. Primary: Product of distances to all-zeros and all-twos vectors.
       Vectors geometrically central in the ternary grid are less likely to
       form 3-term affine lines with boundary vectors.
    2. Secondary: Smooth quadratic distinct-symbol score. Quadratic in the number
       of present symbols gives a smooth landscape that strongly rewards vectors
       using all three coordinate values.
    3. Tertiary: Symmetry-breaking weight favoring 2s to prevent clustering in
       the {0,1}^n sub-cube, forcing uniform exploration of the full space.
    4. Quaternary: Quadratic balance penalty (smooth L2 surrogate for L1).
       Penalizes deviation from uniform (n/3, n/3, n/3) symbol distribution.
    5. Quinary: Variance-based danger proxy. High coordinate variance means more
       extreme coordinates, structurally more likely to complete affine lines.
    6. Quinary: Polynomial rolling hash for deterministic tie-breaking.

    All components computed in O(n) time with minimal per-call overhead.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    # Fast symbol counting in one pass (collects counts, sum, and sum-of-squares)
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

    # Factor 1: Product of distances to all-zeros and all-twos (center distance)
    # dist_to_zero: L1 distance from all-zeros vector
    # dist_to_two: L1 distance from all-twos vector
    # Product rewards vectors that are far from both boundaries (geometric center)
    dist_to_zero = c1 + 2 * c2
    dist_to_two = 2 * c0 + c1
    center_score = dist_to_zero * dist_to_two * 1.0

    # Factor 2: Smooth quadratic distinct-symbol score
    # Quadratic in presence count: 0->0, 1->10000, 2->40000, 3->90000
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 3: Symmetry-breaking weight favoring 2s
    # Prevents clustering in {0,1}^n sub-cube by slightly preferring vectors with 2s
    weight_score = c2 * 100.0

    # Factor 4: Quadratic balance penalty (smooth surrogate for L1)
    # L2 deviation from the uniform (n/3, n/3, n/3) distribution
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # Factor 5: Variance-based danger proxy
    # High coordinate variance = more extreme coordinates = higher affine line risk
    if n > 0:
        mean = s / n
        variance = s2 / n - mean * mean
        danger_score = -variance * 5000.0
    else:
        danger_score = 0.0

    # Factor 6: Polynomial rolling hash for deterministic tie-breaking
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # Combined priority: higher is added first
    total = center_score + distinct_score + weight_score + balance_score + danger_score + hash_score
    return float(total)
