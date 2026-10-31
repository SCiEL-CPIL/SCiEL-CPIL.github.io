"""Cap set greedy builder with optimized mod-3 line-avoidance and diversity hash."""
import time

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin


def priority(vector, n):
    """
    Priority for greedy cap set construction in {0,1,2}^n.

    Structural design:
    1. Strong distinct symbol score (smooth: 1->10000, 2->40000, 3->90000).
    2. Aggressive Mod-3 coordinate-sum penalty (sum == 0 mod 3 is highly dangerous).
    3. Variance-based danger proxy (high variance = extreme = line risk).
    4. High-weight polynomial rolling hash for deterministic symmetry-breaking.

    Optimized for low per-call overhead.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    # Fast symbol counting and sum calculation in one pass
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
        else: # v == 2
            c2 += 1
            s += 2
            s2 += 4

    # Factor 1: Quadratic distinct symbol score
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Aggressive Mod-3 sum penalty
    # Vectors with sum == 0 mod 3 are the most dangerous for forming lines.
    sum_mod = s % 3
    if sum_mod == 0:
        sum_score = -100000.0
    elif sum_mod == 1:
        sum_score = 5000.0
    else:
        sum_score = 5000.0

    # Factor 3: Variance-based danger proxy
    # High variance indicates the vector is far from the "center" of the cube,
    # which correlates with higher line-completion probability in dense sets.
    mean = s / n
    variance = s2 / n - mean * mean
    danger_score = -variance * 10000.0

    # Factor 4: High-weight hash for symmetry breaking
    # This ensures that among vectors with similar statistical properties,
    # we pick a diverse subset to avoid local clustering.
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = (h / 0x7FFFFFFF) * 1000.0

    # Combined priority
    total = distinct_score + sum_score + danger_score + hash_score
    return float(total)
