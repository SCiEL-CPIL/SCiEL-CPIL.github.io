"""Program for kissing number in dimension 11 using randomized greedy integer vector search."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import random

def construct():
    # Constraint: min pairwise dist >= max norm.
    # We will use vectors with norm squared 4 (norm 2) to keep it simple and dense.
    # Condition: dot(u, v) <= 2 for all u, v in set.
    
    # Generate candidate pool
    candidates = []
    
    # Type A: 2 * e_i
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * s
            candidates.append(v)
            
    # Type B: 4 entries +/-1
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)
            
    # Type C: 5 entries +/-1 (Norm sqrt(5) ~ 2.23)
    # If we mix norms, the constraint is ||u-v||^2 >= max(||u||^2, ||v||^2).
    # Let's stick to norm 2 for now to ensure validity is simple (dot <= 2).
    # But wait, if we use norm 2 vectors only, dot <= 2 is the condition.
    
    # To improve the greedy result, we will use a randomized greedy approach.
    # We will shuffle the candidates and greedily add them if they satisfy the constraint
    # with all already selected vectors.
    # We will repeat this process a few times with different shuffles and keep the best result.
    
    best_vectors = []
    best_count = 0
    
    # Limit the number of restarts to fit in time (120s).
    # Each pass is O(N_candidates * N_selected * D).
    # N_candidates ~ 22 + 5280 = 5302.
    # N_selected might be ~ 500.
    # 5302 * 500 * 11 ~ 29 million ops per pass.
    # 10 passes ~ 290 million ops. Should be fine in < 120s.
    
    for _ in range(20):
        random.shuffle(candidates)
        current_vectors = []
        current_arr = np.empty((0, D), dtype=np.int64)
        
        for v in candidates:
            # Check against current set
            if len(current_vectors) == 0:
                current_vectors.append(v)
                current_arr = v.reshape(1, -1)
            else:
                dots = current_arr @ v
                if np.all(dots <= 2):
                    current_vectors.append(v)
                    current_arr = np.vstack([current_arr, v.reshape(1, -1)])
                    
        if len(current_vectors) > best_count:
            best_count = len(current_vectors)
            best_vectors = current_vectors
            # Early stopping if we hit a known high bound? 
            # Max kissing number in D=11 is 556.
            if best_count >= 556:
                break

    return np.array(best_vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
