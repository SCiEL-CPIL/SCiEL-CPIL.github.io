"""Cap set greedy priority function."""
# EVOLVE-BLOCK-START
import time

_state = {
    'pair_sums': None,
    'last_n': None,
    'start_time': None,
    'deadline': None,
}

def _reset_state(n):
    """Reset state for a new n or new evaluation."""
    _state['pair_sums'] = {}
    _state['last_n'] = n
    _state['start_time'] = time.time()
    _state['deadline'] = _state['start_time'] + 280.0  # Safety margin before 330s kill

def _check_deadline():
    """Check if we're approaching the deadline."""
    if _state['deadline'] is not None and time.time() >= _state['deadline']:
        return True
    return False

def _update_pair_sum(vector, n):
    """
    Update pair-sum tracking for a newly added vector.
    Returns the number of existing vectors that would form a 3-term AP with this vector.
    """
    pair_sums = _state['pair_sums']
    conflict_count = 0
    
    # For each existing vector u, check if (u + v) / 2 exists in the set
    # In F_3^n, three vectors a, b, c form an AP if a + c = 2b
    # So for new vector v, check if for any existing u, (u + v) * inv2 exists
    # inv2 in F_3 is 2 (since 2*2 = 4 = 1 mod 3)
    
    for u in _state['pair_sums'].keys():
        # Compute w = (u + v) * 2 mod 3 (this is the midpoint check)
        # If w exists in pair_sums, then u, w, v form an AP
        w = tuple((u[i] + vector[i]) * 2 % 3 for i in range(n))
        if w in _state['pair_sums']:
            conflict_count += 1
    
    # Add this vector to pair_sums
    _state['pair_sums'][tuple(vector)] = True
    
    return conflict_count

def priority(vector, n):
    """
    Assign priority to a vector in {0,1,2}^n for greedy cap set construction.
    
    Combined heuristic:
    1. Primary: minimize line-completion risk (number of existing pairs that would form AP)
    2. Secondary: prefer balanced coordinate distributions
    3. Tertiary: prefer vectors with fewer 1s (symmetry breaking)
    """
    # Reset state if n changes or on first call
    if _state['last_n'] != n or _state['pair_sums'] is None:
        _reset_state(n)
    
    # Deadline guard: if we're close to timeout, return a safe default
    if _check_deadline():
        # Return a neutral priority to avoid expensive computation
        return 0.0
    
    vec = tuple(vector)
    
    # Check if this vector is already in the set (shouldn't happen in greedy, but safe)
    if vec in _state['pair_sums']:
        return -1e9  # Very low priority for duplicates
    
    # Compute line-completion risk
    conflict_count = _count_conflicts(vec, n)
    
    # Balanced distribution heuristic
    count_0 = 0
    count_1 = 0
    count_2 = 0
    for v in vec:
        if v == 0:
            count_0 += 1
        elif v == 1:
            count_1 += 1
        elif v == 2:
            count_2 += 1
    
    min_count = min(count_0, count_1, count_2)
    imbalance = abs(count_0 - count_1) + abs(count_1 - count_2) + abs(count_0 - count_2)
    
    # Combined score: heavily penalize conflicts, use balance as tiebreaker
    # The conflict count should dominate since it directly affects cap set validity
    base_score = -float(conflict_count) * 10.0
    balance_score = float(min_count) - 0.01 * float(imbalance)
    
    return base_score + balance_score

def _count_conflicts(vec, n):
    """
    Count how many existing vectors u would form a 3-term AP with vec.
    A 3-term AP in F_3^n: a, b, c where a + c = 2b.
    For new vec as one endpoint, check if midpoint exists for each existing u.
    """
    pair_sums = _state['pair_sums']
    if not pair_sums:
        return 0
    
    conflict_count = 0
    
    for u in pair_sums.keys():
        # Midpoint m = (u + vec) * 2 mod 3
        # If m is in the set, then u, m, vec form an AP
        m = tuple((u[i] + vec[i]) * 2 % 3 for i in range(n))
        if m in pair_sums:
            conflict_count += 1
    
    return conflict_count
# EVOLVE-BLOCK-END
