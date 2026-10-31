"""Cap set greedy builder with pair-sum line-avoidance and structural heuristics."""
import time

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin

# Global state for tracking forbidden points (line completions)
_forbidden = set()
_added = set()
_current_n = None


def _init_state(n):
    global _forbidden, _added, _current_n, _start_time
    if _current_n != n:
        _forbidden = set()
        _added = set()
        _current_n = n
        _start_time = time.time()
    elif _start_time is None:
        _start_time = time.time()


def _check_deadline():
    global _start_time
    if _start_time is None:
        return False
    return time.time() - _start_time > _DEADLINE


def priority(vector, n):
    """
    Priority for greedy cap set construction in {0,1,2}^n.
    
    Uses:
    1. Pair-sum tracking to detect and penalize vectors that would complete a line.
    2. Distinct symbol count bonus.
    3. Mod-3 sum penalty.
    4. Variance-based danger proxy.
    5. Deterministic hash for symmetry breaking.
    """
    global _forbidden, _added, _current_n, _start_time
    
    _init_state(n)
    
    if _check_deadline():
        return 0.0
    
    vec_key = tuple(vector)
    
    # Check if this vector is forbidden (would complete a line with two existing points)
    is_forbidden = vec_key in _forbidden
    
    # Fast symbol counting and sum calculation
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
        else:
            c2 += 1
            s += 2
            s2 += 4
    
    # Factor 1: Distinct symbol score
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct
    
    # Factor 2: Mod-3 sum penalty
    sum_mod = s % 3
    if sum_mod == 0:
        sum_score = -50000.0
    else:
        sum_score = 3000.0
    
    # Factor 3: Variance-based danger proxy
    mean = s / n
    variance = s2 / n - mean * mean
    danger_score = -variance * 8000.0
    
    # Factor 4: Hash for symmetry breaking
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = (h / 0x7FFFFFFF) * 500.0
    
    # Factor 5: Forbidden penalty (strong)
    if is_forbidden:
        forbidden_penalty = -200000.0
    else:
        forbidden_penalty = 0.0
    
    total = distinct_score + sum_score + danger_score + hash_score + forbidden_penalty
    
    # If this vector is chosen (high priority), update forbidden set
    # We approximate by checking if priority is very high
    if total > 5000.0 and vec_key not in _added:
        _added.add(vec_key)
        # Update forbidden: for each existing point a, new forbidden point is -a - vector mod 3
        new_forbidden = []
        for a in _added:
            # c = -a - vector mod 3 = (2 - a_i) - vector_i mod 3
            c = tuple((2 - a_i - vector[i]) % 3 for i in range(n))
            if c not in _forbidden and c not in _added:
                new_forbidden.append(c)
        for c in new_forbidden:
            _forbidden.add(c)
    
    return float(total)
