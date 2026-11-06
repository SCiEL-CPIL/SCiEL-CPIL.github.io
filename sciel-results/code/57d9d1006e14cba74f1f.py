import time
import random
import numpy as np

# Global cache: n -> list of vectors in optimal greedy order
_cache = {}
_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _generate_all_vectors(n):
    """Generate all 3^n vectors as a numpy array."""
    if n == 0:
        return np.array([[]], dtype=np.int8)
    vecs = np.array(np.meshgrid(*([np.array([0, 1, 2]) for _ in range(n)])), dtype=np.int8).T.reshape(-1, n)
    return vecs

def _precompute_features(vecs):
    """Precompute features for all vectors."""
    n0 = np.sum(vecs == 0, axis=1).astype(np.float32)
    n1 = np.sum(vecs == 1, axis=1).astype(np.float32)
    n2 = np.sum(vecs == 2, axis=1).astype(np.float32)
    s = np.sum(vecs, axis=1).astype(np.float32)
    # Range: max - min
    rng = (np.max(vecs, axis=1) - np.min(vecs, axis=1)).astype(np.float32)
    # Variance
    mean = s / vecs.shape[1]
    var = np.mean((vecs - mean[:, np.newaxis])**2, axis=1).astype(np.float32)
    
    return np.column_stack((n0, n1, n2, s, rng, var))

def _greedy_cap_set_size(order_indices, n, vecs):
    """
    Calculate the size of the cap set formed by greedily adding vectors 
    in the order specified by order_indices.
    """
    added = set()
    count = 0
    
    for idx in order_indices:
        v = vecs[idx]
        conflict = False
        for a in added:
            b = (-a - v) % 3
            if b in added:
                if b != a and b != v:
                    conflict = True
                    break
        if not conflict:
            added.add(v)
            count += 1
    return count

def _find_best_order(n, time_limit=280):
    if n in _cache:
        return _cache[n]
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    vecs = _generate_all_vectors(n)
    total_vecs = len(vecs)
    feats = _precompute_features(vecs)
    
    rng = random.Random(42)
    
    best_size = 0
    best_order = None
    best_weights = None
    
    # Simulated Annealing Parameters
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.995
    
    # Initialize weights
    weights = np.array([rng.uniform(-1, 1) for _ in range(6)])
    
    # Initial evaluation
    noise = rng.random(total_vecs) * 0.01
    scores = feats @ weights + noise
    order_indices = np.argsort(scores)[::-1]
    current_size = _greedy_cap_set_size(order_indices, n, vecs)
    
    if current_size > best_size:
        best_size = current_size
        best_order = order_indices.copy()
        best_weights = weights.copy()
    
    while time.time() < deadline:
        # Perturb weights
        i = rng.randint(0, 5)
        delta = rng.gauss(0, 0.1)
        old_w = weights[i]
        weights[i] += delta
        
        scores = feats @ weights + noise
        order_indices = np.argsort(scores)[::-1]
        new_size = _greedy_cap_set_size(order_indices, n, vecs)
        
        delta_E = current_size - new_size # We want to minimize -size, so delta_E = old - new? 
        # If new_size > current_size, delta_E is negative (improvement).
        # Accept if delta_E < 0 (improvement) or with prob exp(-delta_E / T)
        
        if new_size > current_size:
            current_size = new_size
            if new_size > best_size:
                best_size = new_size
                best_order = order_indices.copy()
                best_weights = weights.copy()
        else:
            # Check acceptance probability
            if delta_E > 0:
                prob = np.exp(-delta_E / T)
                if rng.random() < prob:
                    current_size = new_size
                else:
                    # Revert
                    weights[i] = old_w
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Optionally reset T or restart?
            # For now, keep cooling.
            
    _cache[n] = best_order
    return best_order

def priority(vector, n):
    """Return priority for a vector. Higher = added earlier in greedy."""
    if n not in _cache:
        _find_best_order(n, time_limit=280)
    
    order = _cache[n]
    
    if not hasattr(priority, '_index_cache'):
        priority._index_cache = {}
    
    if n not in priority._index_cache:
        vecs = _generate_all_vectors(n)
        # Build rank map: tuple -> rank (0 is best)
        rank_map = {}
        for rank, idx in enumerate(order):
            v_tuple = tuple(vecs[idx])
            rank_map[v_tuple] = rank
            
        priority._index_cache[n] = rank_map
    
    rank_map = priority._index_cache[n]
    rank = rank_map.get(tuple(vector), len(order))
    return float(len(order) - rank)
