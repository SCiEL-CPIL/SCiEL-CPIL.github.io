"""Cap set greedy construction with a structured priority heuristic."""
import time
import hashlib
import random

# EVOLVE-BLOCK-START

_DEADLINE = time.time() + 280  # safety margin before 300s kill

def _cap_set_ok(existing_set, vec, n):
    """
    Check if adding vec to existing_set keeps it a cap set.
    In F_3^n, a set is a cap set if it contains no 3-term AP.
    When adding vec, we need to ensure no new 3-AP involving vec is formed.
    
    A 3-AP involving vec can have:
    1. vec as the middle term: a + c = 2*vec for some a, c in S
       => c = 2*vec - a, so check if (2*vec - a) in S for any a in S
    2. vec as an end term: a + vec = 2*b for some a, b in S (b is middle)
       => a = 2*b - vec, so check if (2*b - vec) in S for any b in S
    
    Both conditions reduce to: for each x in S, check if (2*x - vec) in S.
    If so, then x, vec, and (2*x - vec) form a 3-AP (or vec, x, and (2*x - vec)).
    """
    vec_tuple = tuple(vec)
    for x in existing_set:
        # Check if vec is an end: is there a y in S such that y = 2*x - vec?
        y = tuple((2 * x[i] - vec_tuple[i]) % 3 for i in range(n))
        if y in existing_set:
            return False
        # Check if vec is the middle: is there a y in S such that y = 2*vec - x?
        z = tuple((2 * vec_tuple[i] - x[i]) % 3 for i in range(n))
        if z in existing_set:
            return False
    return True


def _compute_priority(vector, n):
    """
    Priority heuristic for greedy cap set construction.
    
    Key insight: Vectors with more non-zero entries are "more spread out" in the
    space and tend to participate in fewer 3-APs with other vectors. Vectors with
    fewer non-zero entries (more zeros) are "denser" near the origin and tend to
    block more candidates.
    
    Strategy: Prioritize vectors with more non-zero entries, and among those,
    prefer balanced distributions of 1s and 2s over skewed ones.
    """
    if time.time() > _DEADLINE:
        return 0.0
    
    # Count non-zero entries
    nz = 0
    ones = 0
    twos = 0
    for v in vector:
        if v == 1:
            ones += 1
            nz += 1
        elif v == 2:
            twos += 1
            nz += 1
    
    # Primary: more non-zeros = higher priority (less constraining)
    score = nz * 100.0
    
    # Secondary: balance between 1s and 2s (balanced is better)
    # A perfectly balanced vector has ones ≈ twos
    balance_penalty = abs(ones - twos) * 5.0
    score -= balance_penalty
    
    # Tertiary: use a hash-based tiebreaker for deterministic but varied ordering
    # This helps break ties in a way that explores different regions of the space
    vec_hash = int(hashlib.md5(str(vector).encode()).hexdigest()[:8], 16)
    score += (vec_hash % 1000) * 0.001
    
    return score


def priority(vector, n):
    """
    Priority function for greedy cap set construction.
    Returns a float; higher priority means the vector is considered earlier.
    
    Uses a heuristic that favors vectors with:
    1. More non-zero entries (more spread out, less constraining)
    2. Balanced distribution of 1s and 2s
    3. Deterministic hash-based tiebreaking
    """
    if time.time() > _DEADLINE:
        return 0.0
    return _compute_priority(vector, n)

# EVOLVE-BLOCK-END
