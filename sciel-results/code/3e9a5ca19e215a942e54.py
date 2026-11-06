"""Cap set priority function with deadline-aware greedy construction."""
import time
import itertools
from functools import lru_cache

# Deadline guard: stop work if we exceed this fraction of the 330s limit
_DEADLINE_FRACTION = 0.90
_DEADLINE_SECONDS = 330.0 * _DEADLINE_FRACTION  # 297s

# Cache for cap-set membership checks
# For small n, we can precompute; for larger n, we use a set-based approach

def _is_cap_set_membership(vec, existing_vectors, n):
    """Check if adding vec to existing_vectors keeps it a cap set.
    
    A cap set has no three distinct elements a, b, c with a + b + c = 0 mod 3.
    Equivalently, no three distinct elements form an arithmetic progression:
    a + b = 2c mod 3, or a + c = 2b mod 3, or b + c = 2a mod 3.
    
    For each existing pair (v1, v2), we check if v1 + v2 + vec = 0 mod 3,
    i.e., vec = -(v1 + v2) mod 3 = (2*v1 + 2*v2) mod 3 element-wise.
    """
    for i in range(len(existing_vectors)):
        for j in range(i + 1, len(existing_vectors)):
            v1 = existing_vectors[i]
            v2 = existing_vectors[j]
            # Check if v1 + v2 + vec = 0 mod 3
            # i.e., for each coordinate: v1[k] + v2[k] + vec[k] % 3 == 0
            valid = True
            for k in range(n):
                if (v1[k] + v2[k] + vec[k]) % 3 != 0:
                    valid = False
                    break
            if valid:
                return False
    # Also check pairs involving vec and one existing vector
    for i in range(len(existing_vectors)):
        v1 = existing_vectors[i]
        # Check if v1 + vec + vec = 0 mod 3? No, we need three DISTINCT elements.
        # But wait: the cap set condition is that no three DISTINCT elements sum to 0.
        # So we only need to check triples of distinct elements.
        # When adding vec, we check all pairs (v1, v2) from existing, and also
        # pairs (v1, vec) where v1 is existing.
        # For pair (v1, vec): need a third element v3 such that v1 + vec + v3 = 0 mod 3
        # v3 = -(v1 + vec) mod 3
        # But v3 must be in the existing set AND distinct from v1 and vec.
        # Actually, the standard check: for each pair (a, b) in the set including the new element,
        # check if c = -(a+b) mod 3 is also in the set and distinct from a and b.
        
        # Let me redo this more carefully.
        # When we add vec to S, the new set is S ∪ {vec}.
        # We need to check that there are no three distinct elements in S ∪ {vec} summing to 0.
        # Cases:
        # 1. All three in S: already checked (S was a cap set)
        # 2. Two in S, one is vec: for each pair (v1, v2) in S, check if v1+v2+vec=0 mod 3
        # 3. One in S, two are vec: impossible since we need distinct elements
        # 4. All three are vec: impossible since distinct
        # So case 2 is the only new check.
        pass
    
    return True


def _greedy_cap_set_size(n, priority_func, time_limit):
    """Build a cap set greedily by adding vectors in descending priority order.
    
    Returns the size of the cap set found.
    """
    start_time = time.time()
    
    # Generate all vectors in {0,1,2}^n
    # For efficiency, use itertools.product
    all_vectors = list(itertools.product(range(3), repeat=n))
    
    # Sort by priority descending
    all_vectors.sort(key=lambda v: priority_func(v, n), reverse=True)
    
    cap_set = []
    cap_set_set = set()  # For O(1) lookup
    
    for vec in all_vectors:
        # Check deadline
        if time.time() - start_time > time_limit:
            break
        
        # Check if adding vec keeps the set a cap set
        can_add = True
        vec_tuple = tuple(vec)
        
        # Check pairs in cap_set: for each pair (v1, v2), check if v1+v2+vec=0 mod 3
        for i in range(len(cap_set)):
            if time.time() - start_time > time_limit:
                break
            v1 = cap_set[i]
            for j in range(i + 1, len(cap_set)):
                if time.time() - start_time > time_limit:
                    break
                v2 = cap_set[j]
                # Check if v1 + v2 + vec = 0 mod 3
                ok = True
                for k in range(n):
                    if (v1[k] + v2[k] + vec_tuple[k]) % 3 != 0:
                        ok = False
                        break
                if ok:
                    can_add = False
                    break
            if not can_add:
                break
        
        if can_add:
            cap_set.append(vec_tuple)
            cap_set_set.add(vec_tuple)
    
    return len(cap_set)


# For the priority function, we need to be fast.
# The evaluator calls priority(vector, n) to get a priority score.
# The greedy algorithm then sorts by this priority and adds vectors.
# 
# Strategy: Use a hash-based priority that creates a good ordering.
# A good heuristic for cap sets: vectors with more balanced coordinate values
# tend to be more "compatible". But let's use a simpler approach.
#
# Actually, the key insight: the priority function is called for EACH vector,
# and the greedy algorithm runs once. So we want a priority that leads to
# a large cap set when used in greedy insertion.
#
# Known good strategies for cap sets:
# 1. Random ordering (with a fixed seed for determinism)
# 2. Balance-based: prefer vectors with coordinates closer to 1 (the mean)
# 3. Use a hash to create a pseudo-random but deterministic ordering
#
# Let's use a combination: a hash-based priority that gives a good spread,
# combined with a slight bias toward balanced vectors.

def _vector_hash_priority(vec, n):
    """Compute a priority based on a hash of the vector.
    
    We use a simple hash that gives a pseudo-random but deterministic value.
    This avoids local clustering and explores the space uniformly.
    """
    # Convert vector to a base-3 number
    val = 0
    multiplier = 1
    for i in range(n):
        val += vec[i] * multiplier
        multiplier *= 3
    
    # Use a simple hash function (xorshift-like)
    h = val
    h = (h ^ (h >> 16)) & 0xFFFFFFFF
    h = (h * 0x45d9f3b) & 0xFFFFFFFF
    h = (h ^ (h >> 16)) & 0xFFFFFFFF
    h = (h * 0x45d9f3b) & 0xFFFFFFFF
    h = (h ^ (h >> 16)) & 0xFFFFFFFF
    
    return float(h)


def _balanced_priority(vec, n):
    """Priority based on balance of coordinates.
    
    Vectors with coordinates closer to 1 (the mean of {0,1,2}) get higher priority.
    This is because balanced vectors are less likely to form arithmetic progressions.
    """
    # Sum of squared deviations from 1
    score = 0.0
    for i in range(n):
        dev = vec[i] - 1
        score += dev * dev
    # Lower score = more balanced = higher priority
    return -float(score)


def _combined_priority(vec, n):
    """Combined priority: hash + balance.
    
    Use a weighted combination to get a good ordering.
    """
    hash_prio = _vector_hash_priority(vec, n)
    bal_prio = _balanced_priority(vec, n)
    
    # Normalize and combine
    # Hash priority is in [0, 2^32), balance is in [-n, 0]
    # Scale balance to [0, 1]
    if n > 0:
        bal_norm = (bal_prio + n) / n  # [0, 1]
    else:
        bal_norm = 0.0
    
    # Combine: 70% hash, 30% balance
    return 0.7 * hash_prio + 0.3 * bal_norm


def priority(vector, n):
    """Priority function for greedy cap set construction.
    
    Args:
        vector: tuple of length n with values in {0, 1, 2}
        n: dimension
    
    Returns:
        float: priority score (higher = added earlier)
    """
    # Use the combined priority
    return _combined_priority(vector, n)
