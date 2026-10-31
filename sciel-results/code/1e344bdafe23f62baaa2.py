import math
import time
import random

_start_time = None
_DEADLINE_FRACTION = 0.92
_conflict_cache = {}
_seen_vectors = set()
_MAX_CONFLICT_SAMPLE = 200

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy:
    1. Estimate geometric conflict: count how many affine lines (triplets)
       this vector would form with other vectors in the current set.
    2. Among low-conflict vectors, prefer those with 3 distinct symbols.
    3. Among those, prefer balanced symbol counts.
    4. Use a hash for deterministic tie-breaking.
    
    An affine line in {0,1,2}^n is a set {a, b, c} where b is the midpoint
    of a and c: b_i = (a_i + c_i) / 2 for each coordinate. Equivalently,
    for a given vector v, it forms a line with u if there exists w such that
    v = (u + w) / 2, i.e., w = 2v - u.
    """
    global _start_time, _conflict_cache, _seen_vectors, _MAX_CONFLICT_SAMPLE
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > 330 * _DEADLINE_FRACTION:
        # Emergency: return a neutral priority to avoid hanging
        return 0.0

    # Compute symbol counts
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
            
    counts = [c0, c1, c2]
    distinct_count = sum(1 for c in counts if c > 0)
    
    # Balance score
    ideal = n / 3.0
    if ideal > 0:
        chi2 = ((c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2) / ideal
    else:
        chi2 = 0.0
        
    balance_score = -chi2
    
    # Hash for tie-breaking
    h = 0
    base = 31
    for i, val in enumerate(vector):
        h = (h * base + val + i * 7) % (10**9 + 7)
    hash_score = h / (10**9 + 7)
    
    # Geometric conflict estimation
    # A vector v forms a line with u if w = 2v - u is in the set (or u itself is the midpoint)
    # More precisely: v, u, w form an affine line if v = (u+w)/2, u = (v+w)/2, or w = (v+u)/2
    # For a fixed v, we check if there exists u in seen such that 2v - u is also in seen,
    # OR if u is the midpoint: u = (v + w)/2 for some w in seen.
    
    # Simpler approach: count how many pairs (u, w) in seen satisfy v = (u+w)/2
    # i.e., w = 2v - u. For each u in seen, check if 2v - u is in seen.
    
    conflict_count = 0
    if _seen_vectors:
        # Sample if too many vectors to check all
        seen_list = list(_seen_vectors)
        if len(seen_list) > _MAX_CONFLICT_SAMPLE:
            # Use a deterministic sample based on hash
            rng = random.Random(h % (10**9 + 7))
            sample = rng.sample(seen_list, _MAX_CONFLICT_SAMPLE)
        else:
            sample = seen_list
        
        # Precompute 2*vector components
        two_v = [2 * val for val in vector]
        
        for u in sample:
            # Check if w = 2v - u is in seen
            w = tuple(two_v[i] - u[i] for i in range(n))
            if w in _seen_vectors:
                conflict_count += 1
            # Also check if u is the midpoint: u = (v + w)/2 => w = 2u - v
            w2 = tuple(2 * u[i] - vector[i] for i in range(n))
            if w2 in _seen_vectors:
                conflict_count += 1
    
    # Normalize conflict by sample size
    sample_size = min(len(_seen_vectors), _MAX_CONFLICT_SAMPLE)
    if sample_size > 0:
        conflict_density = conflict_count / sample_size
    else:
        conflict_density = 0.0
    
    # The priority should FAVOR low conflict (negative conflict)
    # Scale conflict to be significant but not dominate entirely
    conflict_score = -conflict_density * 5000.0
    
    # Combine scores
    # Distinct count dominates for structural preference
    distinct_score = distinct_count * 10000.0
    balance_scaled = balance_score * 10.0
    
    return float(distinct_score + balance_scaled + conflict_score + hash_score)
