import itertools
import numpy as np
import time

D = 11

# EVOLVE-BLOCK-START
import random

def generate_vectors(weight):
    """Generate all integer vectors of given weight with entries +/-1."""
    vecs = []
    indices = range(D)
    for comb in itertools.combinations(indices, weight):
        # Generate all sign combinations for the selected indices
        for signs in itertools.product([1, -1], repeat=weight):
            v = [0] * D
            for idx, s in zip(comb, signs):
                v[idx] = s
            vecs.append(v)
    return np.array(vecs, dtype=np.int64)

def greedy_select(pool, max_dot_product):
    """
    Greedily select a subset of vectors from pool such that
    pairwise dot products are <= max_dot_product.
    """
    if pool is None or len(pool) == 0:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    selected = []
    # We process in the given order
    for v in pool:
        # Check against all selected
        # v is shape (D,), selected is list of (D,) arrays
        if not selected:
            selected.append(v)
        else:
            # Compute dot products
            # selected_arr = np.array(selected)
            # dots = selected_arr @ v
            # if np.all(dots <= max_dot_product):
            #     selected.append(v)
            
            # Optimization: early exit
            compatible = True
            for u in selected:
                if np.dot(u, v) > max_dot_product:
                    compatible = False
                    break
            if compatible:
                selected.append(v)
                
    return np.array(selected, dtype=np.int64)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_vectors = np.array([], dtype=np.int64).reshape(0, D)
    best_score = 0
    
    # Strategy 1: Weight 3
    # Max dot product allowed: 1
    pool_w3 = generate_vectors(3)
    # Shuffle and try multiple times
    n_restarts_w3 = 50
    for i in range(n_restarts_w3):
        if time.time() > deadline:
            break
        shuffled_pool = pool_w3.copy()
        np.random.shuffle(shuffled_pool)
        # We can also try sorting by some heuristic, but random is good for diversity
        cand = greedy_select(shuffled_pool, 1)
        if len(cand) > best_score:
            best_score = len(cand)
            best_vectors = cand

    # Strategy 2: Weight 4
    # Max dot product allowed: 2
    pool_w4 = generate_vectors(4)
    n_restarts_w4 = 20
    for i in range(n_restarts_w4):
        if time.time() > deadline:
            break
        shuffled_pool = pool_w4.copy()
        np.random.shuffle(shuffled_pool)
        cand = greedy_select(shuffled_pool, 2)
        if len(cand) > best_score:
            best_score = len(cand)
            best_vectors = cand

    # Strategy 3: Weight 2 (Seed)
    # Max dot product allowed: 1 (since norm^2=2, dist^2>=2 -> 4-2dot>=2 -> dot<=1)
    # Seed got 220.
    pool_w2 = generate_vectors(2)
    n_restarts_w2 = 10
    for i in range(n_restarts_w2):
        if time.time() > deadline:
            break
        shuffled_pool = pool_w2.copy()
        np.random.shuffle(shuffled_pool)
        cand = greedy_select(shuffled_pool, 1)
        if len(cand) > best_score:
            best_score = len(cand)
            best_vectors = cand

    # If no vectors found (should not happen), return empty
    if best_vectors.size == 0:
         return np.array([], dtype=np.int64).reshape(0, D)
         
    return best_vectors
# EVOLVE-BLOCK-END
