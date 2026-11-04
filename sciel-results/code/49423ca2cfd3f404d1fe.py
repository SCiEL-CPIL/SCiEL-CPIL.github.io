import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    D = 11
    
    # 1. Generate Type A vectors (Norm 2)
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)
    vectors_A = np.array(vectors_A, dtype=np.int64)
    
    # 2. Generate Type F vectors (Five 1s, Norm sqrt(5))
    # We will select a large subset of these that are mutually compatible
    # Condition: u . v <= 2 for all u, v in set
    # Note: For u=v, dot=5. We must select distinct vectors.
    # For u != v, max dot is 3 (if intersection 3 and signs match) or 4 (if intersection 4 and 1 mismatch? No, max is intersection size).
    # If intersection 4, max dot 4. 4 > 2. So we must avoid pairs with dot > 2.
    
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 5))
    N_S = len(supports)
    
    # Precompute close supports (intersection >= 3)
    # For each support S, find all T such that |S intersect T| >= 3
    support_to_idx = {s: i for i, s in enumerate(supports)}
    close_supports = [[] for _ in range(N_S)]
    
    for i, S in enumerate(supports):
        S_set = set(S)
        for j, T in enumerate(supports):
            if i == j:
                continue
            T_set = set(T)
            inter_len = len(S_set.intersection(T_set))
            if inter_len >= 3:
                close_supports[i].append(j)
                
    # Generate all candidate vectors for Type F
    # We use a greedy approach with random restarts to maximize the set
    # To speed up, we process vectors in random order
    
    rng = np.random.RandomState(42)
    
    best_set = []
    best_size = 0
    
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # Create list of all possible vectors
        # For each support, generate 32 sign combinations
        all_vectors = []
        vector_supports = []
        
        for s_idx, s in enumerate(supports):
            for signs in itertools.product([1, -1], repeat=5):
                v = [0] * D
                for idx, sign in zip(s, signs):
                    v[idx] = sign
                all_vectors.append(v)
                vector_supports.append(s_idx)
                
        all_vectors = np.array(all_vectors, dtype=np.int64)
        vector_supports = np.array(vector_supports, dtype=np.int64)
        
        # Shuffle
        order = np.arange(len(all_vectors))
        rng.shuffle(order)
        
        selected = []
        selected_supports = [[] for _ in range(N_S)] # List of vectors (indices in all_vectors? or the vectors themselves?)
        # Storing vectors themselves is easier for dot product
        
        for idx in order:
            if time.time() > deadline - 10:
                break
                
            v = all_vectors[idx]
            s_idx = vector_supports[idx]
            
            # Check against close supports
            compatible = True
            for t_idx in close_supports[s_idx]:
                if selected_supports[t_idx]:
                    # Stack vectors from this support
                    # Note: We only need to check against selected vectors in close supports
                    # If we store them in a list, we can vectorize
                    pass
            
            # Optimization: Maintain a list of selected vectors grouped by support
            # But checking 181 supports * (avg 15 vectors) = 2700 dot products
            # Let's just collect all vectors from close supports into a temporary array
            
            candidates = []
            for t_idx in close_supports[s_idx]:
                if selected_supports[t_idx]:
                    candidates.extend(selected_supports[t_idx])
                    
            if candidates:
                candidates_arr = np.array(candidates, dtype=np.int64)
                dots = candidates_arr @ v
                if np.any(dots > 2):
                    compatible = False
            
            if compatible:
                selected.append(v)
                selected_supports[s_idx].append(v)
                
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
            
    final_vectors_F = np.array(best_set, dtype=np.int64) if best_set else np.empty((0, D), dtype=np.int64)
    
    # Combine A and F
    # Check if A and F are compatible
    # We proved they are: a . f <= 2 always.
    # And A . A <= 2?
    # a1 = (2,0..), a2 = (0,2..). Dot 0.
    # a1 = (2,0..), a2 = (2,0..). Same.
    # a1 = (2,0..), a2 = (-2,0..). Dot -4.
    # Condition for A vectors: ||a1 - a2|| >= R_max.
    # If we include F, R_max = sqrt(5) ~ 2.23.
    # ||a1 - a2|| for distinct A is sqrt(8) ~ 2.82.
    # 2.82 >= 2.23. True.
    # So all A vectors are compatible with each other and with F.
    
    final_vectors = np.vstack([vectors_A, final_vectors_F])
    
    return final_vectors
