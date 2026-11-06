import time
import random
import math
from functools import lru_cache

# Global cache: n -> (best_weights, best_size, features, hashes, all_vecs)
_cache = {}
_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _generate_all_vectors(n):
    """Generate all 3^n vectors."""
    if n == 0:
        return [()]
    result = []
    def _gen(prefix, depth):
        if depth == n:
            result.append(tuple(prefix))
            return
        for v in range(3):
            prefix.append(v)
            _gen(prefix, depth+1)
            prefix.pop()
    _gen([], 0)
    return result

def _precompute_features(all_vecs):
    """
    Precompute features for each vector.
    Features: count of 0s, count of 1s, count of 2s.
    Also precompute a deterministic pseudo-random hash for tie-breaking.
    """
    features = []
    hashes = []
    # Use a fixed seed for hash to ensure reproducibility within a run
    rng = random.Random(12345)
    
    for v in all_vecs:
        c0 = v.count(0)
        c1 = v.count(1)
        c2 = v.count(2)
        features.append((c0, c1, c2))
        # Hash in [-1, 1]
        h = rng.uniform(-1, 1)
        hashes.append(h)
        
    return features, hashes

def _run_greedy(order, n, all_vecs):
    """
    Given a list of indices in priority order, return the cap set size.
    """
    added_set = set()
    count = 0
    # Local variables for speed
    vecs = all_vecs
    n_range = range(n)
    
    for idx in order:
        v = vecs[idx]
        conflict = False
        # Check if v conflicts with any pair in added_set
        # a + b + v = 0 mod 3 => b = -a - v
        for a in added_set:
            # Calculate b
            b = tuple((-a[i] - v[i]) % 3 for i in n_range)
            if b in added_set:
                # Ensure distinctness
                if b != a and b != v:
                    conflict = True
                    break
        if not conflict:
            added_set.add(v)
            count += 1
    return count

def _evaluate(weights, n, all_vecs, features, hashes):
    """
    Evaluate the cap set size for a given set of weights.
    weights: [w0, w1, w2, w_hash]
    """
    N = len(all_vecs)
    # Calculate scores
    # w0*c0 + w1*c1 + w2*c2 + w_hash*hash
    w0, w1, w2, wh = weights
    scores = [0.0] * N
    
    for i in range(N):
        f = features[i]
        h = hashes[i]
        scores[i] = w0 * f[0] + w1 * f[1] + w2 * f[2] + wh * h
        
    # Sort indices by score descending
    order = sorted(range(N), key=lambda i: scores[i], reverse=True)
    
    # Run greedy
    return _run_greedy(order, n, all_vecs)

def _find_best_weights(n, time_limit=280):
    """
    Find optimal weights using Coordinate Descent with Random Restarts.
    """
    if n in _cache:
        return _cache[n]
    
    all_vecs = _generate_all_vectors(n)
    features, hashes = _precompute_features(all_vecs)
    
    # We will store the best weights found
    best_weights = None
    best_size = 0
    
    # Coordinate Descent / Hill Climbing on 4D space
    # Dimensions: w0, w1, w2, w_hash
    
    rng = random.Random(42)
    
    # Multiple random restarts
    max_restarts = 10
    time_start = time.time()
    
    for restart in range(max_restarts):
        if time.time() - time_start > time_limit:
            break
            
        # Initialize weights randomly
        # Scale: weights for counts should be comparable.
        # Hash weight should be small to act as tie-breaker, but large enough to break symmetries.
        w0 = rng.uniform(-1.0, 1.0)
        w1 = rng.uniform(-1.0, 1.0)
        w2 = rng.uniform(-1.0, 1.0)
        wh = rng.uniform(-0.1, 0.1) # Start with small hash weight
        
        current_weights = [w0, w1, w2, wh]
        current_size = _evaluate(current_weights, n, all_vecs, features, hashes)
        
        if current_size > best_size:
            best_size = current_size
            best_weights = current_weights[:]
            
        # Local search: Coordinate descent
        improved = True
        step_size = 0.2
        
        while improved and (time.time() - time_start < time_limit):
            improved = False
            # Try moving each dimension
            for dim in range(4):
                # Try positive and negative steps
                for delta in [step_size, -step_size]:
                    new_weights = current_weights[:]
                    new_weights[dim] += delta
                    new_size = _evaluate(new_weights, n, all_vecs, features, hashes)
                    
                    if new_size > current_size:
                        current_size = new_size
                        current_weights = new_weights
                        improved = True
                        if current_size > best_size:
                            best_size = current_size
                            best_weights = current_weights[:]
                        # Reduce step size slightly if we move? Or keep constant.
                        # Keep constant for now, but maybe shrink if stuck?
                        # Let's shrink step size on improvement to zoom in
                        step_size *= 0.8
                        if step_size < 0.01:
                            step_size = 0.01
                    # If no improvement in this direction, try the other direction?
                    # The loop handles both.
            
            # If no improvement, shrink step size further or break?
            # If step size is already small, break.
            if not improved:
                step_size *= 0.5
                if step_size < 0.005:
                    break
    
    _cache[n] = (best_weights, best_size, all_vecs, features, hashes)
    return _cache[n]

def priority(vector, n):
    """Return priority for a vector. Higher = added earlier in greedy."""
    if n not in _cache:
        _find_best_weights(n, time_limit=280)
    
    best_weights, best_size, all_vecs, features, hashes = _cache[n]
    
    # We need to map vector to its features and hash.
    # Since we generated all_vecs in a deterministic order, we can find the index.
    # However, doing a linear search for every priority call is O(3^n).
    # We should precompute a map from vector to index.
    
    if not hasattr(priority, '_vec_to_idx'):
        priority._vec_to_idx = {}
        
    if n not in priority._vec_to_idx:
        # Build map
        priority._vec_to_idx[n] = {v: i for i, v in enumerate(all_vecs)}
        
    idx_map = priority._vec_to_idx[n]
    idx = idx_map[vector]
    
    # Calculate score using best weights
    w0, w1, w2, wh = best_weights
    f = features[idx]
    h = hashes[idx]
    
    score = w0 * f[0] + w1 * f[1] + w2 * f[2] + wh * h
    return float(score)
