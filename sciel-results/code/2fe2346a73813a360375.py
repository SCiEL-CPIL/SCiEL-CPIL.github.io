import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin for 150s limit
    
    best_vectors = []
    best_score = 0

    # Helper to generate all vectors with exactly 'n_nonzero' non-zero entries of value +/-1
    def generate_vectors(n_nonzero):
        vectors = []
        # Choose indices
        indices = list(itertools.combinations(range(D), n_nonzero))
        # Choose signs
        signs = list(itertools.product([1, -1], repeat=n_nonzero))
        for idx in indices:
            for s in signs:
                v = [0] * D
                for i, sign in zip(idx, s):
                    v[i] = sign
                vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    # Helper to run greedy selection for a given set of candidates and max allowed dot product
    # Condition: min_dist >= max_norm
    # For vectors of norm R, min_dist^2 = 2R^2 - 2u.v >= R^2 => 2u.v <= R^2 => u.v <= R^2/2
    def greedy_select(candidates, max_dot):
        if len(candidates) == 0:
            return np.array([], dtype=np.int64).reshape(0, D)
        
        selected = []
        # We can speed this up by checking against the selected set
        # To avoid O(N^2) in pure Python, we use NumPy for dot products
        # But adding one by one is necessary to update the selected set
        
        # Optimization: If N is large, this might be slow.
        # N for norm 2 is 5280.
        # N for norm sqrt(3) is 1320.
        # N for norm sqrt(2) is 220.
        
        # Let's try a simple greedy
        # To make it faster, we can randomize the order or just iterate
        # Iterating in order is fine for now.
        
        for v in candidates:
            if selected:
                sel = np.array(selected)
                dots = sel @ v
                if np.max(dots) <= max_dot:
                    selected.append(v)
            else:
                selected.append(v)
                
            if time.time() > deadline:
                break
                
        return np.array(selected, dtype=np.int64)

    # 1. Try Norm sqrt(2) (2 non-zeros)
    # R^2 = 2. max_dot = 2/2 = 1.
    # Wait, for seed: u.v for (1,1) and (1,-1) is 0. For (1,1) and (0,0,1,1) is 0.
    # Max dot product for valid seed pairs is 0?
    # (1,1,0..) and (1,1,0..) is same vector.
    # (1,1,0..) and (1,0,1..) dot is 1.
    # Distance squared: 2+2 - 2(1) = 2. Distance sqrt(2).
    # Max norm sqrt(2). Condition: dist >= sqrt(2).
    # So dot product <= 1 is correct.
    
    cand_sqrt2 = generate_vectors(2)
    res_sqrt2 = greedy_select(cand_sqrt2, 1)
    if len(res_sqrt2) > best_score:
        best_score = len(res_sqrt2)
        best_vectors = res_sqrt2

    if time.time() > deadline:
        return best_vectors if best_score > 0 else np.array([], dtype=np.int64).reshape(0, D)

    # 2. Try Norm sqrt(3) (3 non-zeros)
    # R^2 = 3. max_dot = 3/2 = 1.5 -> 1.
    cand_sqrt3 = generate_vectors(3)
    res_sqrt3 = greedy_select(cand_sqrt3, 1)
    if len(res_sqrt3) > best_score:
        best_score = len(res_sqrt3)
        best_vectors = res_sqrt3

    if time.time() > deadline:
        return best_vectors if best_score > 0 else np.array([], dtype=np.int64).reshape(0, D)

    # 3. Try Norm 2 (4 non-zeros)
    # R^2 = 4. max_dot = 4/2 = 2.
    cand_2 = generate_vectors(4)
    res_2 = greedy_select(cand_2, 2)
    if len(res_2) > best_score:
        best_score = len(res_2)
        best_vectors = res_2

    # 4. Try Norm sqrt(5) (5 non-zeros)
    # R^2 = 5. max_dot = 5/2 = 2.5 -> 2.
    # Count: C(11,5)*32 = 462*32 = 14784. Might be slow.
    # Let's check time first.
    if time.time() < deadline - 30:
        cand_sqrt5 = generate_vectors(5)
        res_sqrt5 = greedy_select(cand_sqrt5, 2)
        if len(res_sqrt5) > best_score:
            best_score = len(res_sqrt5)
            best_vectors = res_sqrt5

    return best_vectors if best_score > 0 else np.array([], dtype=np.int64).reshape(0, D)
# EVOLVE-BLOCK-END
