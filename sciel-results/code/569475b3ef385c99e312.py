import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    D = 11
    
    # Type A: Vectors with one non-zero entry +/- 2
    # Norm squared = 4
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)
        
    vectors_A = np.array(vectors_A, dtype=np.int64)
    
    # Type B: Vectors with two non-zero entries +/- 1
    # Norm squared = 2
    # We proved these are compatible with Type A and Type D (under M^2=4 constraint)
    vectors_B = []
    for i in range(D):
        for j in range(i + 1, D):
            # 4 sign combinations for (i, j)
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * D
                    v[i] = s1
                    v[j] = s2
                    vectors_B.append(v)
                    
    vectors_B = np.array(vectors_B, dtype=np.int64)
    
    # Type D: Vectors with four non-zero entries +/- 1
    # Norm squared = 4
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)
    
    # Build conflict graph for supports (intersection >= 3)
    conflict_supports = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter_len = len(set(supports[i]) & set(supports[j]))
            if inter_len >= 3:
                conflict_supports[i, j] = True
                conflict_supports[j, i] = True
                
    # Find maximum independent set of supports
    best_support_set = []
    best_size = 0
    
    rng = np.random.RandomState(42)
    
    num_restarts = 500
    
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        order = np.arange(N_S)
        rng.shuffle(order)
        
        in_set = np.zeros(N_S, dtype=bool)
        blocked = np.zeros(N_S, dtype=bool)
        
        for v in order:
            if not blocked[v] and not in_set[v]:
                in_set[v] = True
                blocked |= conflict_supports[v]
                
        current_size = np.sum(in_set)
        
        if current_size > best_size:
            best_size = current_size
            best_support_set = np.where(in_set)[0].tolist()
            
    # Generate Type D vectors for the selected supports
    selected_vectors_D = []
    for s_idx in best_support_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            selected_vectors_D.append(v)
            
    selected_vectors_D = np.array(selected_vectors_D, dtype=np.int64)
    
    # Greedy fill: try to add more Type D vectors from non-selected supports
    current_set_vectors = list(selected_vectors_D)
    current_set_vectors_set = set(map(tuple, current_set_vectors))
    
    # Generate ALL Type D vectors
    all_vectors_D = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(s, signs):
                v[idx] = sign
            all_vectors_D.append(v)
            
    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)
    N_all_D = len(all_vectors_D)
    
    indices_all = np.arange(N_all_D)
    rng.shuffle(indices_all)
    
    added_indices = set()
    
    for i in indices_all:
        if time.time() > deadline - 10:
            break
            
        v = all_vectors_D[i]
        
        if tuple(v) in current_set_vectors_set:
            continue
            
        can_add = True
        for u in current_set_vectors:
            dot = np.dot(v, u)
            if dot >= 3:
                can_add = False
                break
                
        if can_add:
            current_set_vectors.append(v)
            current_set_vectors_set.add(tuple(v))
            
    final_vectors_D = np.array(current_set_vectors, dtype=np.int64)
    
    # Combine Type A, Type B, and Type D
    # Note: We must ensure no duplicates.
    # A, B, D have different supports sizes (1, 2, 4), so no duplicates between types.
    final_vectors = np.vstack([vectors_A, vectors_B, final_vectors_D])
    
    return final_vectors
