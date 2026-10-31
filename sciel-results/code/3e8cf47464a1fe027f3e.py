import time
from collections import Counter
import itertools

_START = time.time()
_DEADLINE = _START + 280.0

_cache = {}

def _generate_vectors(n):
    return [tuple(v) for v in itertools.product([0, 1, 2], repeat=n)]

def _base_score(vector, n):
    c = Counter(vector)
    c0 = c.get(0, 0)
    c1 = c.get(1, 0)
    c2 = c.get(2, 0)
    
    distinct_count = sum(1 for c_val in [c0, c1, c2] if c_val > 0)
    sum_sq = c1 + 4 * c2
    
    ideal = n / 3.0
    dev = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    
    # Hamming distance to center vector (1,1,...,1)
    # Encourages spatial spread away from the center
    hamming_from_center = c0 + c2
    
    # Higher distinct count is better (uses more symbols)
    # Higher sum_sq is better (favors 2s over 0s, breaks symmetry)
    # Lower deviation is better (balanced symbol usage)
    # Higher Hamming distance from center encourages spread
    return distinct_count * 10000.0 + sum_sq * 100.0 - dev * 10.0 + hamming_from_center * 50.0

def _compute_penalties(vectors, base_scores, n):
    """
    Compute penalty for each vector based on its tendency to form lines
    with other high-priority vectors. For a fixed v, a line is formed with
    u and w = -v-u (mod 3). We penalize v proportionally to the minimum
    base score of its line partners, as lower-scored partners are less
    likely to be selected, reducing the actual conflict.
    """
    num_vecs = len(vectors)
    penalties = [0.0] * num_vecs
    vec_to_idx = {v: i for i, v in enumerate(vectors)}
    
    for i, v in enumerate(vectors):
        total_pen = 0.0
        for j, u in enumerate(vectors):
            if i == j:
                continue
            # Compute w = -v - u mod 3
            w = tuple((3 - (v[k] + u[k]) % 3) % 3 for k in range(n))
            if w == u:
                # Not a distinct triplet (u == w means 2u = -v, i.e., u = 2v)
                continue
            k = vec_to_idx.get(w)
            if k is None:
                continue
            bu = base_scores[j]
            bw = base_scores[k]
            # Each pair {u, w} is counted twice (once for u, once for w)
            # So we use 0.5 * min(bu, bw) as the penalty contribution
            total_pen += 0.5 * min(bu, bw)
        penalties[i] = total_pen
        
    return penalties

def _precompute(n):
    if time.time() - _START > _DEADLINE - 10.0:
        return None
    
    vectors = _generate_vectors(n)
    base_scores = [_base_score(v, n) for v in vectors]
    
    if n <= 8:
        penalties = _compute_penalties(vectors, base_scores, n)
        # Normalize penalty by number of potential partners
        num_pairs = max(1, (len(vectors) - 1) / 2.0)
        avg_penalties = [p / num_pairs for p in penalties]
        
        # Combine: base score minus normalized penalty
        # This rewards vectors that are intrinsically good AND don't
        # form lines with other high-priority vectors
        final_scores = [b - p for b, p in zip(base_scores, avg_penalties)]
    else:
        # For large n, skip penalty computation (too expensive)
        final_scores = base_scores
        
    score_map = {v: s for v, s in zip(vectors, final_scores)}
    return score_map

def priority(vector, n):
    """
    Return a float priority for a vector in {0,1,2}^n.
    """
    if time.time() - _START > _DEADLINE:
        return 0.0
    
    if n not in _cache:
        if time.time() - _START > _DEADLINE - 60.0:
            _cache[n] = None
        else:
            _cache[n] = _precompute(n)
            
    if _cache[n] is None:
        # Fallback: base score only
        return _base_score(vector, n)
        
    return _cache[n][vector]
