import time
import random

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin, trusting the directive's 330s kill limit

# Cache for random vectors keyed by n
_random_vectors = {}

def _get_random_vector(n):
    if n not in _random_vectors:
        # Use a deterministic seed based on n to ensure reproducibility
        # but "random" structure relative to the grid
        rng = random.Random(n)
        _random_vectors[n] = [rng.choice([-1, 1]) for _ in range(n)]
    return _random_vectors[n]

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.

    Strategy (Refined Smooth Surrogate Heuristic + Random Projection):
    1. Primary: Random linear projection (chaotic non-local ordering).
    2. Secondary: Smooth quadratic distinct-symbol score.
    3. Tertiary: Mod-3 sum penalty (line-avoidance proxy).
    4. Quaternary: Smooth variance-based danger proxy.
    5. Quinary: Quadratic balance penalty.
    6. Senary: Symmetry breaking via count of 2s.
    7. Septenary: Fast polynomial rolling hash for deterministic tie-breaking.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    # Fast symbol counting and mod-3 sum in one pass
    c0 = 0
    c1 = 0
    c2 = 0
    s = 0
    s2 = 0
    mod3_sum = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
            s += 1
            s2 += 1
            mod3_sum += 1
        else:  # v == 2
            c2 += 1
            s += 2
            s2 += 4
            mod3_sum += 2
    mod3_sum %= 3

    # Factor 1: Random linear projection (primary signal for chaotic ordering)
    R = _get_random_vector(n)
    dot_product = 0
    for i, val in enumerate(vector):
        dot_product += val * R[i]
    random_score = dot_product * 50.0  # Increased weight to be primary signal

    # Factor 2: Smooth quadratic distinct-symbol score.
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 3: Mod-3 sum penalty (vectors with sum ≡ 0 mod 3 are line-prone)
    # Smooth penalty: penalize proximity to 0 mod 3
    mod3_penalty = -500.0 if mod3_sum == 0 else -100.0 if mod3_sum == 1 else 0.0

    # Factor 4: Smooth variance-based danger proxy.
    mean = s / n
    variance = s2 / n - mean * mean
    danger_score = -variance * 5000.0

    # Factor 5: Quadratic balance penalty.
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # Factor 6: Symmetry breaking.
    weight_score = c2 * 100.0

    # Factor 7: Fast polynomial rolling hash for deterministic tie-breaking.
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # Combined priority: higher is added first.
    total = distinct_score + random_score + mod3_penalty + danger_score + balance_score + weight_score + hash_score
    return float(total)
