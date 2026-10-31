"""Cap set construction via greedy selection with geometric-center and hash priorities."""
import time

# EVOLVE-BLOCK-START

_start_time = time.time()
_DEADLINE = 325  # seconds, safety margin before 330s kill

# Precomputed primes for fast hashing
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
           73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
           157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
           239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317,
           331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419,
           421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503]

def _fast_hash(vector, n, seed=0):
    """Fast deterministic pseudo-random priority via linear prime-based hash."""
    h = 0
    for i, v in enumerate(vector):
        if i < len(_PRIMES):
            h += v * _PRIMES[i]
        else:
            h += v * (i * 31 + seed + 17)
    # Mix with seed and reduce to [0, 1)
    h = (h * 1000003 + seed * 999983) % (1 << 60)
    return h / (1 << 60)

def _geometric_center_score(vector):
    """
    Product of distances to all-zeros and all-twos vectors.
    Vectors with many 1s and balanced 0s/2s are prioritized because they're
    geometrically central and less likely to form 3-term APs with boundary vectors.
    """
    count_0 = 0
    count_1 = 0
    count_2 = 0
    for v in vector:
        if v == 0:
            count_0 += 1
        elif v == 1:
            count_1 += 1
        else:
            count_2 += 1
    
    # Product of distances to all-zeros and all-twos
    dist_to_zero = count_1 + 2 * count_2
    dist_to_two = 2 * count_0 + count_1
    center_score = dist_to_zero * dist_to_two
    
    # Diversity bonus: reward having all three values present
    diversity_bonus = 0.0
    if count_0 > 0 and count_1 > 0 and count_2 > 0:
        # Scale bonus by how balanced the distribution is
        min_count = min(count_0, count_1, count_2)
        diversity_bonus = min_count * 2.0
    elif count_1 > 0 and (count_0 > 0 or count_2 > 0):
        # Partial bonus for having 1s and at least one other type
        diversity_bonus = 0.5 * min(count_0 + count_2, count_1)
    
    return center_score + diversity_bonus

def _structured_priority(vector, n):
    """
    Combined priority: geometric center score dominates, with hash-based
    diversity for tie-breaking and robustness.
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        return -1e18
    
    # Geometric center score (dominant signal)
    geo_score = _geometric_center_score(vector)
    
    # Balance: prefer n1 and n2 to be close to each other
    n1 = 0
    n2 = 0
    for v in vector:
        if v == 1:
            n1 += 1
        elif v == 2:
            n2 += 1
    
    balance = abs(n1 - n2) / max(1, n)
    balance_score = (1.0 - balance) * 0.3
    
    # Multiple hash layers for robustness (fast)
    h1 = _fast_hash(vector, n, seed=1)
    h2 = _fast_hash(vector, n, seed=2)
    h3 = _fast_hash(vector, n, seed=3)
    
    # Hash component provides diversity and tie-breaking
    hash_score = (h1 * 0.4 + h2 * 0.35 + h3 * 0.25) * 0.5
    
    # Combine: geometric center dominates, balance and hash provide refinement
    return geo_score + balance_score + hash_score

def priority(vector, n):
    """
    Priority function for greedy cap set construction.
    Higher priority = selected earlier.
    """
    return _structured_priority(vector, n)

# EVOLVE-BLOCK-END
