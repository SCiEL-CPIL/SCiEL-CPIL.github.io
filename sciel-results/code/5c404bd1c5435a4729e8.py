"""Cap set greedy builder: hybrid structural + chaos-breaking priority.

This program combines the "line-blocking potential" structural signal from the
current program with the "chaos-breaking" symmetry-breaking and smooth
quadratic distinctness signals from the top-performing inspirations.

The core insight is that the current program's structural signal (center_dist,
entropy) is too local and symmetric, leading to the greedy algorithm getting
trapped in low-complexity sub-cubes (like {0,1}^n). By adding a strong
symmetry-breaking term (favoring vectors with more 2s) and a smooth quadratic
distinctness score, we force the greedy selection to explore the full {0,1,2}^n
space more uniformly while still respecting the local geometric constraints
imposed by the line-blocking potential.

We also refine the danger proxy by using a smooth variance penalty instead of
the discrete sum-of-squares center distance, and add a fast polynomial hash
for robust tie-breaking to avoid degeneracies in the priority landscape.
"""
import time

_start_time = None
_n_cache = None
_priority_cache = None
_deadline = 320.0  # Safety margin before the 330s kill


def _init(n):
    global _start_time, _n_cache, _priority_cache
    if _start_time is None:
        _start_time = time.time()
    if _n_cache is None or _n_cache != n:
        _n_cache = n
        _priority_cache = {}


def _hybrid_priority(vec, n):
    """
    Deterministic, vector-intrinsic hybrid priority.

    Components:
    1. distinct_score: Smooth quadratic score for using all 3 symbols.
    2. weight_score: Symmetry-breaking bonus for vectors with more 2s.
    3. balance_score: L2 penalty for deviation from uniform symbol distribution.
    4. danger_score: Smooth variance penalty (high variance = more line risk).
    5. blocking_score: Structural line-blocking potential (center + entropy).
    6. hash_score: Fast polynomial hash for deterministic tie-breaking.
    """
    # --- Fast symbol counting in one pass ---
    c0 = 0
    c1 = 0
    c2 = 0
    s = 0
    s2 = 0
    for v in vec:
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

    total = n
    if total == 0:
        return 0.0

    # Factor 1: Smooth quadratic distinct-symbol score.
    # Quadratic in presence count: 0 -> 0, 1 -> 10000, 2 -> 40000, 3 -> 90000.
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Symmetry breaking via count of 2s.
    # Favors vectors with more 2s to break the symmetry of the ternary grid
    # and prevent the greedy algorithm from clustering in the {0,1}^n sub-cube.
    weight_score = c2 * 100.0

    # Factor 3: Quadratic balance penalty (smooth surrogate for L1).
    # L2 deviation from the uniform (n/3, n/3, n/3) distribution.
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # Factor 4: Smooth variance-based danger proxy.
    # High coordinate variance = more extreme coordinates = higher line risk.
    mean = s / total
    variance = s2 / total - mean * mean
    danger_score = -variance * 5000.0

    # Factor 5: Structural line-blocking potential (from current program).
    # We retain the entropy and center-distance signals but re-weight them
    # to be secondary to the chaos-breaking terms.
    import numpy as np
    entropy = 0.0
    for cnt in (c0, c1, c2):
        if cnt > 0:
            p = cnt / total
            entropy -= p * np.log(p + 1e-15)
    ent_norm = entropy / np.log(3.0)

    # Center distance: 1 at center (all 1s), 0 at corners (all 0s or 2s).
    # sum_sq = s2. Center is at sum_sq = 2*n.
    max_sum_sq = 4 * n
    center_dist = 1.0 - abs(s2 - 2 * n) / max_sum_sq

    blocking_score = 1.0 * center_dist + 0.7 * ent_norm

    # Factor 6: Fast polynomial rolling hash for deterministic tie-breaking.
    h = 0
    for i, val in enumerate(vec):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # Factor 7: Multiplicative complexity score.
    # Treat 0 as 1 for multiplicative stability. The product of (val+1) values
    # rewards vectors with more 2s and penalizes vectors with many 0s,
    # enforcing a "full-rank" algebraic structure that resists additive
    # dependencies forming affine lines.
    # Product of (v+1) for each coordinate: 0->1, 1->2, 2->3
    # Log of product = sum of log(v+1), which is smooth and avoids overflow.
    import math
    log_product = 0.0
    for v in vec:
        log_product += math.log(v + 1)
    # Normalize by n to get average log-factor per coordinate
    # Range: [0, log(3)] per coordinate, so total range [0, n*log(3)]
    # Scale to be competitive with other terms
    mult_score = log_product * 15.0

    # --- Combine ---
    # The distinct_score and weight_score are the dominant "chaos-breaking" signals.
    # The danger_score and balance_score are the smooth "safety" signals.
    # The blocking_score is the structural "line-blocking" signal.
    # The hash_score is a tiny tie-breaker.
    
    # We scale the blocking_score down slightly so that the global symmetry-breaking
    # terms dominate the local structural terms, preventing the greedy algorithm
    # from getting stuck in local optima defined by the lattice symmetries.
    total_priority = (
        distinct_score
        + weight_score
        + balance_score
        + danger_score
        + 50.0 * blocking_score  # Scale up blocking to be competitive but not dominant
        + mult_score
        + hash_score
    )
    
    return float(total_priority)


def priority(vector, n):
    """
    Returns a priority value for the given vector. Higher priority means the
    vector is considered first in the greedy insertion.

    Strategy: Hybrid of chaos-breaking (symmetry breaking + distinctness)
    and structural line-blocking potential.
    """
    _init(n)

    # Deadline guard: return a fast deterministic fallback near the limit.
    elapsed = time.time() - _start_time
    if elapsed > _deadline:
        vec = tuple(vector)
        # Fast fallback: hash + multiplicative proxy for distinctness
        h = 0
        c2 = 0
        for i, val in enumerate(vec):
            h = (h * 31 + val + i) & 0x7FFFFFFF
            if val == 2:
                c2 += 1
        return float(h + c2 * 100.0)

    vec = tuple(vector)

    if vec in _priority_cache:
        return _priority_cache[vec]

    val = _hybrid_priority(vec, n)
    _priority_cache[vec] = val
    return val
