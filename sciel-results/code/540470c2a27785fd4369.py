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
    
    # Hamming distance to the "center" vector (1,1,...,1)
    hamming_from_center = c0 + c2
    
    # Spectral energy: smooth measure of deviation from center
    spectral_energy = 0.0
    for val in vector:
        d = val - 1
        spectral_energy += d * d * (1.0 + 0.1 * d * d)
    
    # Additional: count how many coordinates are "balanced" (one of each 0,1,2 in a window)
    # This is a proxy for local structure that might form lines
    local_balance = 0
    for i in range(n - 2):
        if vector[i] != vector[i+1] != vector[i+2] != vector[i]:
            local_balance += 1
    
    return distinct_count * 10000.0 + sum_sq * 100.0 - dev * 10.0 + hamming_from_center * 50.0 - spectral_energy * 20.0 + local_balance * 5.0

def _soft_min(a, b, tau=1.0):
    """Smooth approximation to min(a, b) using log-sum-exp."""
    m = min(a, b)
    if m == 0:
        return 0.0
    # soft-min = -tau * log(1 + exp(-(a - m)/tau) * exp(-(b - m)/tau) ... )
    # Simpler: soft-min(a,b) = m - tau * log(1 + exp(-(a-m)/tau) + exp(-(b-m)/tau) - 1)
    # Even simpler approximation: m - tau * log(1 + exp(-max(a-m, b-m)/tau))
    # Use: soft_min ≈ m - tau * log(1 + exp(-(max(a,b)-m)/tau))
    # This is a smooth lower bound on min.
    diff = max(a, b) - m
    if diff < 1e-10:
        return m
    return m - tau * math.log1p(math.exp(-diff / tau))

def _compute_penalties(vectors, base_scores, n):
    num_vecs = len(vectors)
    penalties = [0.0] * num_vecs
    # Map vector to index
    vec_to_idx = {v: i for i, v in enumerate(vectors)}
    
    # Soft-min temperature
    tau = 500.0
    
    for i, v in enumerate(vectors):
        total_pen = 0.0
        pen_count = 0
        
        for j, u in enumerate(vectors):
            if i == j: continue
            
            # Compute w = -v - u mod 3
            w = tuple((3 - (v[k] + u[k]) % 3) % 3 for k in range(n))
            if w == u: continue
            
            k = vec_to_idx.get(w)
            if k is None:
                continue
            
            bu = base_scores[j]
            bw = base_scores[k]
            # Use soft-min for smoother landscape
            total_pen += 0.5 * _soft_min(bu, bw, tau)
            pen_count += 1
        
        # Normalize by number of potential line partners
        if pen_count > 0:
            penalties[i] = total_pen / pen_count
        else:
            penalties[i] = 0.0
        
    return penalties

def _precompute(n):
    if time.time() - _START > _DEADLINE - 10.0:
        return None # Timeout risk
        
    vectors = _generate_vectors(n)
    base_scores = [_base_score(v, n) for v in vectors]
    
    if n <= 8:
        penalties = _compute_penalties(vectors, base_scores, n)
        
        # Combine: Base score minus penalty
        # The penalty is already normalized by pen_count in _compute_penalties
        final_scores = [b - p * 0.5 for b, p in zip(base_scores, penalties)]
        
        # Apply a smooth rank-based boost to break ties
        if len(final_scores) > 1:
            max_s = max(final_scores)
            min_s = min(final_scores)
            range_s = max_s - min_s
            if range_s > 1e-10:
                for idx in range(len(final_scores)):
                    norm = (final_scores[idx] - min_s) / range_s
                    # Gentle boost: adds up to 2% of range for top vectors
                    final_scores[idx] += 0.02 * range_s * (norm ** 3)
    else:
        # For large n, skip penalty
        final_scores = base_scores
        
    score_map = {v: s for v, s in zip(vectors, final_scores)}
    return score_map

def priority(vector, n):
    """
    Return a float priority for a vector in {0,1,2}^n.
    """
    # Deadline check
    if time.time() - _START > _DEADLINE:
        return 0.0
    
    if n not in _cache:
        # Pre-compute if not already done
        # Check time before starting heavy pre-computation
        if time.time() - _START > _DEADLINE - 60.0:
            # Not enough time for pre-computation, use fallback
            _cache[n] = None # Mark as fallback
        else:
            _cache[n] = _precompute(n)
            
    if _cache[n] is None:
        # Fallback to simple heuristic
        return _base_score(vector, n)
        
    return _cache[n][vector]
