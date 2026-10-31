# EVOLVE-BLOCK-START
import time

_start_time = None
_deadline = 320.0  # safety margin before the 330s kill

def priority(vector, n):
    """
    Priority for greedy cap-set construction in {0,1,2}^n.

    Structural Strategy:
    1. Quadratic Distinctness: Strongly prefer vectors using all 3 symbols.
       Smooth quadratic scaling creates a steep but continuous gradient.
    2. L2 Balance Penalty: Penalize deviation from uniform symbol distribution
       using squared error (smooth surrogate for balance).
    3. Extreme-Coordinate Penalty: Penalize coordinates at 0 or 2 (extremes)
       which are more likely to complete affine lines. Uses a smooth
       quadratic measure of "extremeness" per coordinate.
    4. Symmetry Breaking: Favor vectors with more 2s to prevent clustering
       in the {0,1}^n sub-cube.
    5. Deterministic Hash: Fast polynomial hash for tie-breaking.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _deadline:
        return 0.0

    # Fast symbol counting in one pass
    c0 = 0
    c1 = 0
    c2 = 0
    extreme_count = 0  # count of coordinates that are 0 or 2
    for v in vector:
        if v == 0:
            c0 += 1
            extreme_count += 1
        elif v == 1:
            c1 += 1
        else:  # v == 2
            c2 += 1
            extreme_count += 1

    # Factor 1: Smooth quadratic distinct-symbol score.
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Symmetry breaking via count of 2s.
    weight_score = c2 * 100.0

    # Factor 3: Quadratic balance penalty.
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # Factor 4: Extreme-coordinate penalty.
    # Each coordinate at 0 or 2 contributes to line-completion risk.
    # Use a quadratic penalty on the fraction of extreme coordinates.
    # This is smoother than raw variance and directly targets the structural
    # property (extreme values) that creates affine lines.
    extreme_frac = extreme_count / n
    extreme_score = -extreme_frac * extreme_frac * 8000.0

    # Factor 5: Fast polynomial rolling hash for deterministic tie-breaking.
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # Combined priority: higher is added first.
    total = distinct_score + weight_score + balance_score + extreme_score + hash_score
    return float(total)
# EVOLVE-BLOCK-END
