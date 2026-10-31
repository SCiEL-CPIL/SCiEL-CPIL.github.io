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

Refinements:
- Precompute log and pair-interaction tables at module load to eliminate
  per-call import overhead and per-element math.log calls.
- Strengthen the smooth balance penalty so the L2 deviation from uniform
  symbol distribution is more competitive with the danger term.
- Add a smooth pair-interaction term that penalizes adjacent-coordinate
  patterns (a, b) where 2b == a + c mod 3 for a third symbol c, a smooth
  surrogate for local arithmetic-progression risk in F_3^n.
"""
import time
import math
import numpy as np

# Precomputed lookup tables for speed
_LOG_TABLE = [0.0, math.log(2.0), math.log(3.0)]  # log(v+1) for v in {0,1,2}
# Pair interaction: for adjacent coords (a,b), penalty if they could form an AP
# In F_3, a triple (x,y,z) is an AP iff x+z = 2y mod 3.
# For a pair (a,b), the "danger" is how easily a third coordinate c exists
# such that {a,b,c} forms an AP. We use a smooth proxy:
# If a==b, then c=a completes a trivial AP (penalty high).
# If a!=b, then c = (2b - a) % 3 or c = (2a - b) % 3 completes an AP.
# We assign a smooth penalty based on the "closeness" in F_3.
_PAIR_PENALTY = [
    [0.0, 0.5, 1.0],  # a=0: b=0 (trivial AP risk), b=1 (mild), b=2 (high)
    [0.5, 0.0, 0.5],  # a=1: b=0 (mild), b=1 (trivial), b=2 (mild)
    [1.0, 0.5, 0.0],  # a=2: b=0 (high), b=1 (mild), b=2 (trivial)
]
# Actually, in F_3 any two distinct elements determine a unique third that
# completes an AP. So all pairs have AP risk. The distinction is:
# - Same value (a==b): the AP is (a,a,a), trivial/degenerate
# - Different values: the AP is non-trivial
# We want to penalize non-trivial APs more since they indicate real structure.
# Let's redefine: penalty is 0 for same, 1.0 for different (all different pairs
# in F_3 complete an AP).
_PAIR_PENALTY = [
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
]

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
    7. mult_score: Log-product complexity score.
    8. pair_score: Smooth penalty for adjacent-coordinate AP risk.
    """
    # --- Fast symbol counting + pair interaction in one pass ---
    c0 = 0
    c1 = 0
    c2 = 0
    s = 0
    s2 = 0
    log_product = 0.0
    pair_penalty = 0.0

    if n > 0:
        prev = vec[0]
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
            log_product += _LOG_TABLE[v]
            if n > 1:
                pair_penalty += _PAIR_PENALTY[prev][v]
                prev = v
        # Adjust: we counted the first element's pair with itself implicitly? No,
        # we start prev = vec[0] and then for each v (including vec[0]) we add
        # _PAIR_PENALTY[prev][v]. For the first iteration, prev=vec[0], v=vec[0],
        # so we get _PAIR_PENALTY[vec[0]][vec[0]] = 0.0. That's fine.
        # But we should only count pairs (i, i+1) for i in 0..n-2.
        # Let me redo: we want sum over i=0 to n-2 of _PAIR_PENALTY[vec[i]][vec[i+1]]
        # The loop above does: for each v in vec, add _PAIR_PENALTY[prev][v] where
        # prev is the previous element (starting with vec[0]).
        # Iteration 0: prev=vec[0], v=vec[0] -> penalty[vec[0]][vec[0]] = 0
        # Iteration 1: prev=vec[0], v=vec[1] -> penalty[vec[0]][vec[1]]
        # ...
        # Iteration n-1: prev=vec[n-2], v=vec[n-1] -> penalty[vec[n-2]][vec[n-1]]
        # So we get exactly the sum of adjacent pairs. Good.

    total = n
    if total == 0:
        return 0.0

    # Factor 1: Smooth quadratic distinct-symbol score.
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Symmetry breaking via count of 2s.
    weight_score = c2 * 100.0

    # Factor 3: Quadratic balance penalty (strengthened from 5.0 to 15.0).
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 15.0

    # Factor 4: Smooth variance-based danger proxy.
    mean = s / total
    variance = s2 / total - mean * mean
    danger_score = -variance * 5000.0

    # Factor 5: Structural line-blocking potential.
    entropy = 0.0
    for cnt in (c0, c1, c2):
        if cnt > 0:
            p = cnt / total
            entropy -= p * math.log(p + 1e-15)
    ent_norm = entropy / math.log(3.0)

    max_sum_sq = 4 * n
    center_dist = 1.0 - abs(s2 - 2 * n) / max_sum_sq

    blocking_score = 1.0 * center_dist + 0.7 * ent_norm

    # Factor 6: Fast polynomial rolling hash for deterministic tie-breaking.
    h = 0
    for i, val in enumerate(vec):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # Factor 7: Multiplicative complexity score (using precomputed log table).
    mult_score = log_product * 15.0

    # Factor 8: Pair-interaction penalty (smooth AP risk surrogate).
    # Scale: max pair_penalty is (n-1) * 1.0. Scale to be a minor signal.
    pair_score = -pair_penalty * 2.0

    # --- Combine ---
    total_priority = (
        distinct_score
        + weight_score
        + balance_score
        + danger_score
        + 50.0 * blocking_score
        + mult_score
        + pair_score
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
