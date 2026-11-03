"""Kissing number in dimension 11 - integer vector construction."""
import numpy as np
import time
import itertools
import random

D = 11
DEADLINE = 145.0  # safety margin before 150s kill


def _pairwise_min_dist_sq(vectors):
    """Compute minimum pairwise squared distance among vectors."""
    if len(vectors) < 2:
        return float('inf')
    # Use matrix multiplication for efficiency
    # min_dist^2 = min_{i!=j} ||v_i - v_j||^2 = min_{i!=j} (||v_i||^2 + ||v_j||^2 - 2 v_i.v_j)
    norms_sq = np.sum(vectors * vectors, axis=1)
    dot = vectors @ vectors.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dot
    np.fill_diagonal(dist_sq, float('inf'))
    return int(dist_sq.min())


def _max_norm_sq(vectors):
    """Maximum squared norm among vectors."""
    if len(vectors) == 0:
        return 0
    return int(np.max(np.sum(vectors * vectors, axis=1)))


def _is_valid(vectors):
    """Check if configuration is valid: min pairwise dist >= max norm."""
    if len(vectors) < 2:
        return True
    min_d2 = _pairwise_min_dist_sq(vectors)
    max_n2 = _max_norm_sq(vectors)
    return min_d2 >= max_n2


def _generate_weight2_vectors():
    """Generate all weight-2 vectors with entries in {-1, 1}."""
    vecs = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vecs.append(tuple(v))
    return vecs


def _generate_weight3_vectors():
    """Generate all weight-3 vectors with entries in {-1, 1}."""
    vecs = []
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = si, sj, sk
                    vecs.append(tuple(v))
    return vecs


def _generate_weight4_vectors():
    """Generate all weight-4 vectors with entries in {-1, 1}."""
    vecs = []
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            vecs.append(tuple(v))
    return vecs


def _greedy_add(vectors, candidates, rng):
    """Greedily add candidates that maintain validity. Returns (vectors, added_count)."""
    added = 0
    for c in candidates:
        if time.time() > DEADLINE:
            break
        test = vectors + [c]
        if _is_valid(test):
            vectors.append(c)
            added += 1
    return vectors, added


def _local_search(vectors, all_candidates, rng, max_iterations=5000):
    """Local search: try swapping out one vector for another to increase count or maintain."""
    current = list(vectors)
    current_set = set(current)
    best = len(current)
    best_vectors = list(current)
    
    # Pre-compute norms
    norms = {v: sum(x*x for x in v) for v in current}
    max_norm = max(norms.values()) if current else 0
    
    for iteration in range(max_iterations):
        if time.time() > DEADLINE:
            break
        
        # Try adding a random candidate
        if rng.random() < 0.7:
            # Try to add a new vector
            available = [c for c in all_candidates if c not in current_set]
            if not available:
                continue
            c = rng.choice(available)
            test = current + [c]
            if _is_valid(test):
                current.append(c)
                current_set.add(c)
                best = max(best, len(current))
                best_vectors = list(current)
        else:
            # Try swapping: remove one, add one
            if len(current) < 2:
                continue
            i = rng.randrange(len(current))
            removed = current[i]
            current_no_i = current[:i] + current[i+1:]
            
            available = [c for c in all_candidates if c not in current_set or c == removed]
            if not available:
                continue
            c = rng.choice(available)
            if c == removed:
                continue
            test = current_no_i + [c]
            if _is_valid(test):
                current = test
                current_set.discard(removed)
                current_set.add(c)
                if len(current) > best:
                    best = len(current)
                    best_vectors = list(current)
    
    return best_vectors, best


def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    # Generate candidate pools
    w2 = _generate_weight2_vectors()
    w3 = _generate_weight3_vectors()
    w4 = _generate_weight4_vectors()
    
    # Strategy 1: Start with all weight-2, greedily add weight-3, then weight-4
    vectors = [list(v) for v in w2]
    # Shuffle for diversity
    rng.shuffle(w3)
    rng.shuffle(w4)
    
    # Greedy add weight-3
    vectors, _ = _greedy_add(vectors, w3, rng)
    n_after_w3 = len(vectors)
    
    # Greedy add weight-4
    vectors, _ = _greedy_add(vectors, w4, rng)
    n_after_w4 = len(vectors)
    
    # Local search to improve
    all_cands = w2 + w3 + w4
    vectors, best_n = _local_search(vectors, all_cands, rng, max_iterations=3000)
    
    # Try different starting points
    # Strategy 2: Start with weight-2 + weight-3, then add weight-4
    if time.time() < DEADLINE - 10:
        vectors2 = [list(v) for v in w2]
        rng2 = random.Random(123)
        w3_shuffled = list(w3)
        rng2.shuffle(w3_shuffled)
        w4_shuffled = list(w4)
        rng2.shuffle(w4_shuffled)
        
        # Add a subset of weight-3 first
        vectors2, _ = _greedy_add(vectors2, w3_shuffled[:2000], rng2)
        vectors2, _ = _greedy_add(vectors2, w4_shuffled, rng2)
        
        if len(vectors2) > len(vectors):
            vectors = vectors2
        
        # Local search
        vectors, best_n = _local_search(vectors, all_cands, rng2, max_iterations=2000)
    
    # Strategy 3: Pure greedy with different orderings
    if time.time() < DEADLINE - 10:
        for seed in [7, 99, 256]:
            if time.time() > DEADLINE - 5:
                break
            rng3 = random.Random(seed)
            w2_shuffled = list(w2)
            w3_shuffled = list(w3)
            w4_shuffled = list(w4)
            rng3.shuffle(w2_shuffled)
            rng3.shuffle(w3_shuffled)
            rng3.shuffle(w4_shuffled)
            
            v3 = []
            # Add all weight-2 in shuffled order
            for v in w2_shuffled:
                if time.time() > DEADLINE - 5:
                    break
                test = v3 + [list(v)]
                if _is_valid(test):
                    v3.append(list(v))
            
            # Add weight-3
            v3, _ = _greedy_add(v3, w3_shuffled, rng3)
            # Add weight-4
            v3, _ = _greedy_add(v3, w4_shuffled, rng3)
            
            if len(v3) > len(vectors):
                vectors = v3
                best_n = len(vectors)
            
            # Quick local search
            v3, n3 = _local_search(v3, all_cands, rng3, max_iterations=1000)
            if n3 > best_n:
                vectors = v3
                best_n = n3
    
    # Final verification
    vectors = [list(v) for v in vectors]
    if not _is_valid(vectors):
        # Fallback: use just weight-2
        vectors = [list(v) for v in w2]
    
    return np.array(vectors, dtype=np.int64)
