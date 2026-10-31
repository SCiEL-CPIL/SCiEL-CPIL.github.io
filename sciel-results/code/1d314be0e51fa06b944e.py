import time
import math

_start_time = None
_DEADLINE = 330 * 0.92  # safety margin

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.

    Strategy (Refined Combined Surrogate):
    - Primary: Smooth quadratic distinct-symbol score (favors all 3 symbols).
    - Secondary: Smooth variance penalty (favors central vectors).
    - Tertiary: Symmetry breaking via count of 2s (prevents {0,1}^n clustering).
    - Quaternary: Soft-balance penalty (smooth L2 deviation from uniform).
    - Quinary: Deterministic hash for tie-breaking.
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

    # Factor 1: Smooth quadratic distinct-symbol score.
    # num_distinct is 1, 2, or 3.
    # Score: 10000, 40000, 90000 respectively.
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Variance-based danger proxy.
    # Penalizes high variance (extreme vectors).
    # Variance = E[X^2] - (E[X])^2
    mean = s / n
    variance = s2 / n - mean * mean
    danger_score = -variance * 6000.0  # Slightly increased penalty

    # Factor 3: Symmetry breaking via count of 2s.
    # Linear preference for 2s.
    weight_score = c2 * 120.0  # Slightly increased to further break symmetry

    # Factor 4: Quadratic balance penalty.
    # Minimizes L2 distance from uniform distribution.
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 6.0  # Slightly increased

    # Factor 5: Fast polynomial rolling hash for deterministic tie-breaking.
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # Combined priority: higher is added first.
    total = distinct_score + danger_score + weight_score + balance_score + hash_score
    return float(total)
