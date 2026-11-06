"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import hashlib
from itertools import product

_DEADLINE = 300.0
_TIME_MARGIN = 5.0

def _hash_priority(vector, n, seed_offset=0):
    """Generate a priority value based on a hash of the vector."""
    h = hashlib.sha256((vector + (seed_offset,)).encode()).digest()
    return int.from_bytes(h[:8], 'big')

def _is_cap_set(vecs, n):
    """Check if a set of vectors forms a cap set."""
    vecs = list(vecs)
    m = len(vecs)
    for i in range(m):
        for j in range(i + 1, m):
            for k in range(j + 1, m):
                vi, vj, vk = vecs[i], vecs[j], vecs[k]
                # Check if vi + vj + vk is a linear combination with coefficients summing to 3
                # In F_3^n, a cap set has no three distinct points in a line
                # Three points a, b, c are in a line if a + c = 2*b (mod 3), i.e., a + c - 2b = 0
                # Or equivalently: for each coordinate, (a_i + b_i + c_i) mod 3 != 3
                # Actually, in F_3, three points are collinear if a_i + b_i + c_i = 0 mod 3 for all i,
                # but that's not quite right either.
                # The correct condition: a, b, c are in a line iff a + c = 2b, i.e., a - b = b - c (in F_3)
                # So for each coordinate i: (a[i] + c[i] - 2*b[i]) % 3 == 0
                collinear = True
                for coord in range(n):
                    if (vi[coord] + vk[coord] - 2 * vj[coord]) % 3 != 0:
                        collinear = False
                        break
                if collinear:
                    return False
    return True

def _greedy_cap_set_size(n, order, time_limit=None):
    """
    Greedily add vectors from the given order to form a cap set.
    Returns the size of the resulting cap set.
    """
    if time_limit is None:
        time_limit = _DEADLINE - _TIME_MARGIN - time.time()
        if time_limit < 0:
            return 0

    start_time = time.time()
    cap_set = []
    cap_set_set = set()

    for vec in order:
        if time.time() - start_time > time_limit:
            break
        # Check if adding this vector keeps it a cap set
        # We need to check that no three vectors in cap_set + {vec} are collinear
        # This means:
        # 1. vec is not already in cap_set
        # 2. No pair (a, b) in cap_set such that a, b, vec are collinear
        # 3. No pair (a, b) in cap_set such that a, b, vec form a line with vec as middle
        # Actually, we just need to check all triples involving vec
        # But that's O(m^2) per insertion where m is current size

        if vec in cap_set_set:
            continue

        # Check if adding vec creates any line
        valid = True
        for a in cap_set:
            for b in cap_set:
                if a == b:
                    continue
                # Check if a, b, vec are collinear
                # a, b, vec are collinear if there exists t such that vec = a + t*(b-a) in F_3
                # Or: a + vec - 2*b = 0, or a + b - 2*vec = 0, or b + vec - 2*a = 0
                # The three points are collinear if any of these hold
                if (a[0] + vec[0] - 2*b[0]) % 3 == 0 and \
                   (a[1] + vec[1] - 2*b[1]) % 3 == 0 and \
                   all((a[i] + vec[i] - 2*b[i]) % 3 == 0 for i in range(n)):
                    valid = False
                    break
                if (a[0] + b[0] - 2*vec[0]) % 3 == 0 and \
                   all((a[i] + b[i] - 2*vec[i]) % 3 == 0 for i in range(n)):
                    valid = False
                    break
                if (b[0] + vec[0] - 2*a[0]) % 3 == 0 and \
                   all((b[i] + vec[i] - 2*a[i]) % 3 == 0 for i in range(n)):
                    valid = False
                    break
            if not valid:
                break

        if valid:
            cap_set.append(vec)
            cap_set_set.add(vec)

    return len(cap_set)

def _generate_all_vectors(n):
    """Generate all vectors in {0,1,2}^n."""
    return list(product(range(3), repeat=n))

def _search_best_priority(n, time_budget):
    """
    Search for a good static priority ordering.
    Uses multiple random seeds and evaluates the greedy cap set size.
    """
    all_vecs = _generate_all_vectors(n)
    num_vecs = len(all_vecs)

    # Pre-compute some useful structures
    # For fast collinearity checks, we can use a set of "directions"

    best_size = 0
    best_order = list(range(num_vecs))
    best_seed = 0

    start_time = time.time()
    deadline = start_time + time_budget

    # Try multiple random seeds
    num_trials = 50
    for trial in range(num_trials):
        if time.time() > deadline - 10:  # Leave 10s margin
            break

        # Create a random ordering using a hash-based approach
        seed = random.randint(0, 2**32 - 1)
        order = sorted(range(num_vecs), key=lambda i: _hash_priority(all_vecs[i], n, seed))

        # Run greedy with this order
        size = _greedy_cap_set_size(n, [all_vecs[i] for i in order], time_limit=max(0, deadline - time.time()))

        if size > best_size:
            best_size = size
            best_order = order
            best_seed = seed

    return best_size, best_order, all_vecs

def priority(vector, n):
    # Uniform priority: greedy insertion in lexicographic order.
    # We'll use an offline-searched priority ordering.
    # Since the interface requires returning a float for each vector,
    # we need to pre-compute the ordering and return priorities accordingly.

    global _cached_priority_map
    global _cached_n
    global _cached_start_time

    current_time = time.time()

    # Check if we have a cached result for this n
    if '_cached_priority_map' not in dir() or _cached_n != n:
        # We need to compute the priority map
        # But we only have time until _DEADLINE
        time_budget = max(0, _DEADLINE - _TIME_MARGIN - current_time)
        if time_budget <= 0:
            # No time left, use a simple default
            return 0.0

        # For small n, we can do a more thorough search
        if n <= 6:
            best_size, best_order, all_vecs = _search_best_priority(n, time_budget)
            _cached_priority_map = {}
            _cached_n = n
            for rank, vec_idx in enumerate(best_order):
                _cached_priority_map[all_vecs[vec_idx]] = float(rank)
            _cached_start_time = current_time
        else:
            # For larger n, use a hash-based priority with a good seed
            # We'll use a fixed seed that works well
            _cached_priority_map = {}
            _cached_n = n
            all_vecs = _generate_all_vectors(n)
            # Use multiple hash seeds and pick the best one quickly
            best_seed = 42
            # Just use a single hash-based ordering for now
            for vec in all_vecs:
                _cached_priority_map[vec] = _hash_priority(vec, n, 42)
            _cached_start_time = current_time

    # Return the priority for this vector
    if vector in _cached_priority_map:
        return _cached_priority_map[vector]

    # Fallback: compute on the fly
    return _hash_priority(vector, n, 42)
# EVOLVE-BLOCK-END
