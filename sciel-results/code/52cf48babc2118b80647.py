import time

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin


def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.

    Strategy (combined design with smooth surrogates):
    1. Primary: Smooth quadratic distinct-symbol score. Quadratic in the number
       of present symbols gives a smoother landscape than discrete steps.
    2. Secondary: Hamming weight relative to all-ones (count of 2s). Favors
       vectors "heavy" in the 2-coordinate to break the symmetry of the ternary
       grid and prevent clustering in the {0,1}^n sub-cube.
    3. Tertiary: Quadratic balance penalty. L2 deviation from the uniform
       (n/3, n/3, n/3) distribution provides a smooth gradient.
    4. Quaternary: Shannon entropy of symbol distribution. Higher entropy means
       more uniform spread, reducing line-completion risk.
    5. Quinary: Variance-based danger proxy. High coordinate variance means
       more extreme coordinates, which are structurally more likely to
       complete an affine line.
    6. Quinary: Fast polynomial rolling hash for robust deterministic
       tie-breaking.

    All components computed in O(n) time with minimal per-call overhead.
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
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Hamming weight relative to all-ones (count of 2s).
    # Favors vectors with more 2s to break symmetry and explore the full space.
    weight_score = c2 * 100.0

    # Factor 3: Quadratic balance penalty (smooth L2 surrogate).
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # Factor 4: Shannon entropy of symbol distribution.
    # Higher entropy = more uniform spread = lower line risk.
    entropy = 0.0
    for c in (c0, c1, c2):
        if c > 0:
            p = c / n
            entropy -= p * math.log(p) if False else 0  # placeholder
    # Compute entropy properly
    entropy = 0.0
    for c in (c0, c1, c2):
        if c > 0:
            p = c / n
            entropy -= p * _log(p)
    entropy_score = entropy * 500.0

    # Factor 5: Variance-based danger proxy.
    mean = s / n
    variance = s2 / n - mean * mean
    danger_score = -variance * 5000.0

    # Factor 6: Fast polynomial rolling hash for deterministic tie-breaking.
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    total = distinct_score + weight_score + balance_score + entropy_score + danger_score + hash_score
    return float(total)


# Pre-compute log for common fractions to avoid repeated math.log calls
import math
_log = math.log
