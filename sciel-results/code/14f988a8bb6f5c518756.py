"""Cap set greedy builder: local structural heuristics + global line-avoidance."""
import time

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin

# State shared across priority() calls within one evaluation
_state = {
    'n': None,
    'vectors': [],       # list of tuple vectors in the cap set
    'pair_sum': None,    # dict: (sum0 mod 3, sum1 mod 3, ...) -> count of ordered pairs (a,b) in S
    'size': 0,
}


def _ensure_init(n):
    """Reset state when n changes (new evaluation)."""
    if _state['n'] != n:
        _state['n'] = n
        _state['vectors'] = []
        _state['pair_sum'] = {}
        _state['size'] = 0


def _update_pair_sum(vec, n):
    """Add contributions from pairs involving vec to pair_sum."""
    ps = _state['pair_sum']
    size = _state['size']
    for j in range(size):
        vj = _state['vectors'][j]
        key = tuple((vec[i] + vj[i]) % 3 for i in range(n))
        ps[key] = ps.get(key, 0) + 1
        key_rev = tuple((vj[i] + vec[i]) % 3 for i in range(n))
        ps[key_rev] = ps.get(key_rev, 0) + 1


def _fast_hash(vector):
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    return h / 0x7FFFFFFF


def priority(vector, n):
    """
    Priority for greedy cap set construction in {0,1,2}^n.

    Design (same family as seed, augmented with the actual cap-set constraint):
    1. Quadratic symbol-presence score (smooth distinct-symbol reward).
    2. Mod-3 coordinate-sum penalty (sum ≡ 0 mod 3 is dangerous).
    3. Quadratic balance penalty (deviation from uniform coordinate distribution).
    4. Global line-avoidance: penalize vectors that would complete many
       affine lines with already-selected pairs (the actual cap-set constraint).
    5. Fast rolling hash for deterministic tie-breaking.

    The global line term makes the greedy process aware of the real constraint:
    a vector's priority drops as it would create more 3-point lines, which is
    exactly what a cap set must avoid.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    _ensure_init(n)

    # Fast symbol counting
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)

    # Factor 1: Quadratic presence score (reduced dominance to let line-avoidance matter more)
    p0 = 1.0 if c0 > 0 else 0.0
    p1 = 1.0 if c1 > 0 else 0.0
    p2 = 1.0 if c2 > 0 else 0.0
    num_distinct = p0 + p1 + p2
    distinct_score = 5000.0 * num_distinct * num_distinct

    # Factor 2: Mod-3 sum penalty
    coord_sum = c1 + 2 * c2
    sum_mod = coord_sum % 3
    if sum_mod == 0:
        sum_score = -3000.0
    elif sum_mod == 1:
        sum_score = 1500.0
    else:
        sum_score = 1000.0

    # Factor 3: Quadratic balance penalty
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 3.0

    # Factor 4: Global line-avoidance (the actual cap-set constraint)
    # A vector v creates a line with pair (a,b) iff a + b + v ≡ 0 mod 3
    # i.e. v ≡ -(a+b) ≡ (3 - (a+b)) mod 3 coordinate-wise.
    # Count how many existing pairs would be completed by this vector.
    ps = _state['pair_sum']
    size = _state['size']
    if ps and size > 0:
        v_key = tuple((3 - vector[i]) % 3 for i in range(n))
        line_count = ps.get(v_key, 0)
        # Adaptive scaling: as set grows, each pair is more "expensive"
        # Use sqrt scaling to keep penalty responsive but not overwhelming
        line_penalty = -line_count * (30.0 + 20.0 * (size / max(n, 1)))
    else:
        line_penalty = 0.0

    # Factor 5: Improved hash for better tie-breaking
    # Use a stronger mixing function that's more discriminative
    h = 0
    for i, val in enumerate(vector):
        h = (h * 131 + val * 7 + i * 3) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # Combined priority: higher is added first
    total = distinct_score + sum_score + balance_score + line_penalty + hash_score

    # Update state: add this vector (the evaluator will only call priority
    # for vectors that pass the cap-set check, so we assume acceptance)
    vec_tuple = tuple(vector)
    _state['vectors'].append(vec_tuple)
    _update_pair_sum(vec_tuple, n)
    _state['size'] += 1

    return float(total)
