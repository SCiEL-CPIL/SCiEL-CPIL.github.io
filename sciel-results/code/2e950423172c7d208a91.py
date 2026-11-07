import time
import random
from itertools import product as iproduct

# EVOLVE-BLOCK-START

# Precompute prime factorizations for 2..40
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
_NUM_PRIMES = len(_PRIMES)

def _prime_exponents(n):
    """Return tuple of exponents of primes in n."""
    exps = []
    for p in _PRIMES:
        cnt = 0
        while n % p == 0:
            cnt += 1
            n //= p
        exps.append(cnt)
    return tuple(exps)

# Precompute exponent vectors for all candidates 2..40
_EXP_VECTORS = {}
for _n in range(2, 41):
    _EXP_VECTORS[_n] = _prime_exponents(_n)

# Precompute all subset-sum vectors for a given set of numbers
# We'll use a different approach: track the set of all achievable exponent vectors

def _can_add_vector(x, current_vectors):
    """
    Check if adding x (with exponent vector vx) creates a collision.
    A collision occurs if for some a, b in current_vectors, a + vx == b,
    i.e., b - a == vx, or equivalently, there exist a, b in current_vectors
    such that b - a = vx.
    
    More precisely: after adding x, the new set of vectors is V ∪ (V + vx).
    A collision exists if (V + vx) ∩ V ≠ ∅ (excluding the trivial case where
    we're just adding new elements), OR if there are duplicates within V+vx.
    
    Actually, the correct check is: the new set of subset products corresponds to
    vectors V ∪ {v + vx : v ∈ V}. This set has size 2^(k+1) iff all these vectors
    are distinct. A collision exists iff there exist v1, v2 ∈ V such that v1 + vx = v2,
    i.e., v2 - v1 = vx.
    """
    vx = _EXP_VECTORS[x]
    # Check if vx can be expressed as a difference of two vectors in current_vectors
    # i.e., if there exist a, b in current_vectors such that b - a = vx
    # This is equivalent to: for some a in current_vectors, a + vx is also in current_vectors
    for a in current_vectors:
        target = tuple(ai + vxi for ai, vxi in zip(a, vx))
        if target in current_vectors:
            return False
    return True

def _add_to_vectors(current_vectors, x):
    """Add x's exponent vector to the set of achievable vectors."""
    vx = _EXP_VECTORS[x]
    new_vectors = set()
    for v in current_vectors:
        new_vectors.add(tuple(vi + vxi for vi, vxi in zip(v, vx)))
    current_vectors.update(new_vectors)
    return current_vectors

def construct():
    """
    Returns a list of integers in 2..40 with distinct subset products.
    Uses randomized greedy search with prime-exponent vector collision detection.
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s kill
    
    pool = list(range(2, 41))
    best_set = []
    best_len = 0
    
    # Known good 16-element set as fallback
    fallback = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Run multiple randomized greedy passes
    num_passes = 0
    while time.time() < deadline:
        random.shuffle(pool)
        current_set = []
        current_vectors = {(0,) * _NUM_PRIMES}  # Start with the empty product (all zeros)
        
        for x in pool:
            if _can_add_vector(x, current_vectors):
                current_set.append(x)
                _add_to_vectors(current_vectors, x)
        
        if len(current_set) > best_len:
            best_set = current_set.copy()
            best_len = len(best_set)
            if best_len == 18:  # Max possible
                break
        num_passes += 1
        
        # Early termination if we've been running a while and haven't improved recently
        if best_len >= 17 and num_passes > 200:
            break

    # Return the best set found, or fallback if nothing was found
    if best_len < 16:
        return list(fallback)
    
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
