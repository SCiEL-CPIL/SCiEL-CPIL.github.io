import time
from collections import Counter
import itertools
import math

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
    
    # Compute Hamming distance to the "center" vector (1,1,...,1)
    hamming_from_center = c0 + c2
    
    # Higher distinct count is better
    # Higher sum_sq is better
    # Lower deviation is better
    # Higher Hamming distance from center encourages spatial spread
    return distinct_count * 10000.0 + sum_sq * 100.0 - dev * 10.0 + hamming_from_center * 50.0

def _compute_penalties(vectors, base_scores, n):
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
                continue
            k = vec_to_idx.get(w)
            if k is None:
                continue
            bu = base_scores[j]
            bw = base_scores[k]
            total_pen += 0.5 * min(bu, bw)
        penalties[i] = total_pen
        
    return penalties

def _ternary_xor_parity(vector, n, mask):
    """Compute parity of XOR between ternary representation and a fixed mask."""
    # Convert vector to integer in base 3
    val = 0
    for i, v in enumerate(vector):
        val = val * 3 + v
    
    # XOR with mask (treat as integer)
    xor_val = val ^ mask
    
    # Count bits in the result
    bit_count = bin(xor_val).count('1')
    
    return float(bit_count)

def _precompute(n):
    if time.time() - _START > _DEADLINE - 10.0:
        return None
    
    vectors = _generate_vectors(n)
    base_scores = [_base_score(v, n) for v in vectors]
    
    # Create a deterministic mask based on n to ensure reproducibility
    # Use a simple hash-like construction
    mask = 0
    for i in range(n):
        mask = (mask * 3 + (i % 3)) ^ (i * 7 + 13)
    
    parity_scores = [_ternary_xor_parity(v, n, mask) for v in vectors]
    
    if n <= 7:
        penalties = _compute_penalties(vectors, base_scores, n)
        num_pairs = (len(vectors) - 1) / 2.0
        if num_pairs > 0:
            avg_penalties = [p / num_pairs for p in penalties]
        else:
            avg_penalties = [0.0] * len(vectors)
        
        # Combine base score, penalty, and parity
        # Parity adds a small chaotic component to break correlations
        final_scores = [b - p + par * 0.1 for b, p, par in zip(base_scores, avg_penalties, parity_scores)]
    else:
        # For large n, skip penalty but still add parity
        final_scores = [b + par * 0.1 for b, par in zip(base_scores, parity_scores)]
        
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
        # Fallback: base score + simple parity
        mask = 0
        for i in range(n):
            mask = (mask * 3 + (i % 3)) ^ (i * 7 + 13)
        par = _ternary_xor_parity(vector, n, mask)
        return _base_score(vector, n) + par * 0.1
        
    return _cache[n][vector]
