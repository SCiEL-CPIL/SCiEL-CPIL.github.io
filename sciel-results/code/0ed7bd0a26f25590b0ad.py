"""Rewritten program for kissing number in dimension 11."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill

    # Generate all weight-4 vectors with entries in {-1, 0, 1}
    v4_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_list.append(v)

    vectors_v4 = np.array(v4_list, dtype=np.int64)
    N4 = len(vectors_v4)

    # Generate weight-1 vectors with entries in {-2, 0, 2}
    v1_list = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_list.append(v)
    vectors_v1 = np.array(v1_list, dtype=np.int64)
    N1 = len(vectors_v1)

    if N4 == 0:
        return vectors_v1

    # Compute pairwise dot products among v4 vectors
    dots = vectors_v4 @ vectors_v4.T
    # Conflict if dot > 2
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # v1 vectors are compatible with all v4 and other v1s.
    # So we just need to find a maximum independent set in the v4 conflict graph.
    # Then add all v1 vectors.

    rng = np.random.default_rng(42)
    
    best_in_set = np.zeros(N4, dtype=bool)
    best_size = 0
    
    def randomized_greedy():
        in_set = np.zeros(N4, dtype=bool)
        order = rng.permutation(N4)
        for i in order:
            if not np.any(conflicts[i] & in_set):
                in_set[i] = True
        return in_set

    def local_search(in_set, max_time=1.0):
        in_set = in_set.copy()
        current_size = in_set.sum()
        best_size_ls = current_size
        best_in_set_ls = in_set.copy()
        
        conflict_count = (conflicts @ in_set).astype(np.int32)
        
        ls_deadline = time.time() + max_time
        T = 1.0
        cooling = 0.99
        min_T = 0.01
        
        while time.time() < ls_deadline:
            not_in = np.where(~in_set)[0]
            if len(not_in) == 0:
                break
            
            scores = 1 - conflict_count[not_in]
            max_score = np.max(scores)
            cands = not_in[scores == max_score]
            v = cands[rng.integers(len(cands))]
            
            delta = 1 - conflict_count[v]
            
            if delta >= 0 or rng.random() < np.exp(delta / T):
                to_remove = np.where(conflicts[v] & in_set)[0]
                
                for u in to_remove:
                    in_set[u] = False
                    conflict_count[conflicts[u]] -= 1
                
                in_set[v] = True
                conflict_count[conflicts[v]] += 1
                
                current_size += delta
                
                if current_size > best_size_ls:
                    best_size_ls = current_size
                    best_in_set_ls = in_set.copy()
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        return best_in_set_ls, best_size_ls

    n_restarts = 0
    max_restarts = 150
    
    while time.time() < deadline - 5 and n_restarts < max_restarts:
        n_restarts += 1
        in_set = randomized_greedy()
        size = in_set.sum()
        
        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()
        
        # Local search
        ls_time = 0.5
        if size > best_size * 0.95:
            ls_time = 1.0
        ls_result, ls_size = local_search(in_set, max_time=ls_time)
        
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()
            
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    
    if len(result_v4) > 0:
        final_vectors = np.vstack([result_v4, vectors_v1])
    else:
        final_vectors = vectors_v1

    return final_vectors

# EVOLVE-BLOCK-END
