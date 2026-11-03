"""Program for kissing number in dimension 11 using randomized greedy search over mixed-norm integer vectors."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import random

def construct():
    # Generate candidate vectors with squared norm 4 and 8.
    candidates = []
    
    # Type 1: Norm squared 4
    # 1. One entry +/-2
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # 2. Four entries +/-1
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)

    # Type 2: Norm squared 8
    # 1. Two entries +/-2
    for support in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for idx, s in zip(support, signs):
                v[idx] = 2 * s
            candidates.append(v)

    # 2. One entry +/-2, two entries +/-1
    for i in range(D):
        for s1 in (1, -1):
            for support in itertools.combinations([j for j in range(D) if j != i], 2):
                for signs in itertools.product([1, -1], repeat=2):
                    v = [0] * D
                    v[i] = 2 * s1
                    for idx, s in zip(support, signs):
                        v[idx] = s
                    candidates.append(v)

    # 3. Eight entries +/-1
    for support in itertools.combinations(range(D), 8):
        for signs in itertools.product([1, -1], repeat=8):
            v = [0] * D
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)

    candidates = np.array(candidates, dtype=np.int64)
    norms_sq = np.sum(candidates**2, axis=1)
    
    best_vectors = []
    
    # Run randomized greedy multiple times
    num_trials = 10
    best_count = 0
    best_result = None
    
    for _ in range(num_trials):
        # Shuffle candidates
        indices = np.arange(len(candidates))
        np.random.shuffle(indices)
        shuffled_candidates = candidates[indices]
        shuffled_norms_sq = norms_sq[indices]
        
        current_vecs = []
        current_norms_sq = []
        
        for i in range(len(shuffled_candidates)):
            v = shuffled_candidates[i]
            v_norm_sq = shuffled_norms_sq[i]
            
            if not current_vecs:
                current_vecs.append(v)
                current_norms_sq.append(v_norm_sq)
                continue
                
            # Check compatibility with all existing vectors
            # Constraint: ||v - u||^2 >= max(||v||^2, ||u||^2)
            # ||v||^2 + ||u||^2 - 2*v.dot(u) >= max(||v||^2, ||u||^2)
            # If v_norm_sq >= u_norm_sq:
            #   v_norm_sq + u_norm_sq - 2*dot >= v_norm_sq
            #   u_norm_sq - 2*dot >= 0  => dot <= u_norm_sq / 2
            # If u_norm_sq > v_norm_sq:
            #   v_norm_sq + u_norm_sq - 2*dot >= u_norm_sq
            #   v_norm_sq - 2*dot >= 0  => dot <= v_norm_sq / 2
            
            # General condition: dot <= min(v_norm_sq, u_norm_sq) / 2
            
            dots = np.dot(current_vecs, v)
            min_norms_sq = np.minimum(current_norms_sq, v_norm_sq)
            max_allowed_dots = min_norms_sq / 2.0
            
            if np.all(dots <= max_allowed_dots):
                current_vecs.append(v)
                current_norms_sq.append(v_norm_sq)
                
        if len(current_vecs) > best_count:
            best_count = len(current_vecs)
            best_result = np.array(current_vecs)
            
    return best_result
# EVOLVE-BLOCK-END
