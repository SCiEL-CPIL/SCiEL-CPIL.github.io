"""Cap set greedy builder with geometric conflict minimization."""
import time
import itertools
import numpy as np

# We track state across calls within a single evaluation run.
# The evaluator calls priority(vector, n) to rank vectors.
# We maintain a global set of accepted vectors and a cap-set check.

_start_time = None
_accepted = None       # set of tuples
_all_vectors_cache = None
_n_cache = None
_priority_cache = None  # dict mapping vector -> priority value
_deadline = 300.0       # seconds; safety margin
_safety_margin = 10.0   # stop this much before deadline

def _init(n):
    global _start_time, _accepted, _all_vectors_cache, _n_cache, _priority_cache
    if _start_time is None:
        _start_time = time.time()
    if _n_cache is None or _n_cache != n:
        _n_cache = n
        _accepted = set()
        _priority_cache = {}
        # Pre-generate all vectors? For small n this is feasible.
        # For larger n, we can't pre-generate 3^n vectors.
        # We'll check on the fly.
        _all_vectors_cache = None

def _is_cap_set(vecs):
    """Check if a set of vectors in {0,1,2}^n is a cap set (no three collinear).
    Three vectors a, b, c in {0,1,2}^n are collinear iff a + b + c = (0,0,...,0) mod 3
    (i.e., they form an affine line in AG(n,3)).
    Equivalently, for any two distinct vectors a, b in the set, the third point
    c = (-a - b) mod 3 must NOT be in the set.
    """
    vec_list = list(vecs)
    n = len(vec_list[0]) if vec_list else 0
    for i in range(len(vec_list)):
        for j in range(i + 1, len(vec_list)):
            a = vec_list[i]
            b = vec_list[j]
            # c[k] = (-a[k] - b[k]) % 3
            c = tuple((-a[k] - b[k]) % 3 for k in range(n))
            if c in vecs:
                return False
    return True

def _forms_line_with_new(vec, existing_set, n):
    """Check if adding vec to existing_set creates a line.
    For each pair (a, b) in existing_set, check if the third point
    c = (-a - b) % 3 equals vec, OR if vec with a gives b, etc.
    More efficiently: for each a in existing_set, compute c = (-vec - a) % 3.
    If c is in existing_set, then (vec, a, c) is a line.
    """
    for a in existing_set:
        c = tuple((-vec[k] - a[k]) % 3 for k in range(n))
        if c in existing_set:
            return True
    return False

def _compute_conflict_score(vec, existing_set, n):
    """Count how many pairs in existing_set would form a line if we added vec.
    This is the number of a in existing_set such that c = (-vec - a) % 3 is also in existing_set.
    Lower is better (fewer future conflicts).
    """
    count = 0
    for a in existing_set:
        c = tuple((-vec[k] - a[k]) % 3 for k in range(n))
        if c in existing_set:
            count += 1
    return count

def priority(vector, n):
    """
    Returns a priority value for the given vector. Higher priority means
    the vector is considered first in the greedy insertion.
    
    Strategy: Minimize geometric conflict (number of affine lines formed
    with pairs in the current set). Use a smooth surrogate via negative
    conflict count, plus a small perturbation for tie-breaking based on
    coordinate balance.
    """
    _init(n)
    
    # Deadline check
    elapsed = time.time() - _start_time
    if elapsed > _deadline - _safety_margin:
        # Return a deterministic fallback to avoid further computation
        return hash(vector) % 1000000
    
    vec = tuple(vector)
    
    # If already cached, return cached value
    if vec in _priority_cache:
        return _priority_cache[vec]
    
    # If the vector already forms a line with the current set, give it very low priority
    if _forms_line_with_new(vec, _accepted, n):
        _priority_cache[vec] = -1e9
        return _priority_cache[vec]
    
    # Compute conflict score: number of pairs in _accepted that would form
    # a line involving vec. Lower conflict = higher priority.
    conflict = _compute_conflict_score(vec, _accepted, n)
    
    # Add a small balance term: prefer vectors with more uniform coordinate distribution
    # This helps break ties and guides toward "central" vectors
    coord_counts = [0, 0, 0]
    for val in vec:
        coord_counts[val] += 1
    # Entropy-like measure: uniform distribution is better
    total = n
    balance = 0.0
    for cnt in coord_counts:
        if cnt > 0:
            p = cnt / total
            balance -= p * np.log(p + 1e-15)
    
    # Priority: minimize conflict, maximize balance
    # Use a weighted combination
    priority_val = -conflict * 10.0 + balance * 0.1
    
    _priority_cache[vec] = priority_val
    return priority_val

def _accept_vector(vec, n):
    """Accept a vector into the cap set if it doesn't create a line."""
    if not _forms_line_with_new(vec, _accepted, n):
        _accepted.add(vec)
        # Invalidate priority cache for vectors that interact with this one
        # (For efficiency, we could be more selective, but for correctness
        #  we clear the cache when the set changes significantly)
        # Actually, the conflict scores change when we add a vector, so
        #  the cached priorities are stale. We need to handle this.
        # For now, we'll recompute on the fly if the set has changed.
        # A simpler approach: don't cache, or cache with a version number.
        pass

# The evaluator works as follows:
# 1. It generates all vectors in {0,1,2}^n
# 2. It calls priority(vector, n) for each vector
# 3. It sorts vectors by priority in descending order
# 4. It greedily adds vectors in that order if they keep the set a cap set
# 5. The score is the final cap set size
#
# The problem: our priority function depends on the current set, but the
# evaluator computes all priorities upfront before doing the greedy insertion.
# This means we can't use the "current set" in a meaningful way during
# the priority computation phase.
#
# We need a static priority function that doesn't depend on the evolving set.
# Let's redesign: use a static heuristic based on the vector's properties.

_start_time = None

def priority(vector, n):
    """
    Static priority function for cap set construction.
    
    Since the evaluator computes all priorities before greedy insertion,
    we need a vector-intrinsic heuristic. Strategy:
    - Prefer vectors with balanced coordinates (entropy)
    - Use a structured ordering that tends to produce larger cap sets
    - Add a small hash-based perturbation for diversity
    
    The key insight: in {0,1,2}^n, cap sets avoid affine lines.
    Vectors that are "far" from forming lines with many others are better.
    A good static proxy: vectors with coordinates that are not all equal
    and have good spread tend to be part of larger cap sets.
    
    We use a combination of:
    1. Coordinate entropy (uniformity)
    2. Number of distinct coordinates (more distinct = potentially less collinear)
    3. A deterministic hash for tie-breaking
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > 290.0:
        return 0.0
    
    vec = tuple(vector)
    
    # Coordinate balance: entropy of the distribution of 0, 1, 2
    counts = [0, 0, 0]
    for v in vec:
        counts[v] += 1
    
    total = n
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * np.log(p + 1e-15)
    
    # Number of distinct coordinate values (1, 2, or 3)
    distinct = sum(1 for c in counts if c > 0)
    
    # Vectors with all same coordinate (like (0,0,...,0)) are bad
    # because they form lines with many others
    all_same = 1.0 if distinct == 1 else 0.0
    
    # Sum of squares of coordinates (measures "distance from center")
    sum_sq = sum(v * v for v in vec)
    
    # A good cap set tends to include vectors that are "spread out"
    # Use: high entropy + high distinct + low all_same penalty + moderate sum_sq
    
    # Normalize sum_sq to [0, 1]
    max_sum_sq = 4 * n  # max is 4*n when all coords are 2
    norm_sum_sq = sum_sq / max_sum_sq if max_sum_sq > 0 else 0
    
    # Priority: entropy is good, distinct is good, all_same is bad
    # We want vectors that are balanced but not too extreme
    priority_val = entropy * 2.0 + distinct * 0.5 - all_same * 5.0
    
    # Add a small deterministic perturbation based on the vector itself
    # This breaks ties in a structured way
    # Use a simple hash
    h = 0
    for i, v in enumerate(vec):
        h = (h * 31 + v + i) % 1000003
    priority_val += h * 1e-6
    
    return priority_val
