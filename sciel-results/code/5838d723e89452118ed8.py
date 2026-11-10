import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 58.0
    
    n = 12
    m = 40
    total_sets = 1 << n
    
    # Precompute comparability matrix
    # comp[i][j] is 1 if i and j are comparable (one strict subset of other)
    # Diagonal is 0.
    
    j_arr = np.arange(total_sets)
    comp_matrix = np.zeros((total_sets, total_sets), dtype=np.int32)
    
    # Vectorized precomputation
    # For each i, compute row i
    for i in range(total_sets):
        # (j & i) == j  => j subset of i
        # (j & i) == i  => i subset of j
        row = ((j_arr & i) == j_arr) | ((j_arr & i) == i)
        row[i] = 0
        comp_matrix[i] = row.astype(np.int32)
        
    def calc_score_vec(S_vec):
        # S_vec is int array of 0s and 1s
        # Score = 0.5 * sum_{i,j in S} comp[i][j]
        # = 0.5 * (S_vec @ (comp_matrix @ S_vec))
        return 0.5 * (S_vec @ (comp_matrix @ S_vec))
        
    def local_search_numpy(S_vec, time_limit=None):
        # S_vec: np.array of int32 (0 or 1)
        current_score = calc_score_vec(S_vec)
        best_score = current_score
        best_S_vec = S_vec.copy()
        
        # G[i] = sum of comp[i][j] for j in S
        G = (comp_matrix @ S_vec).astype(np.int32)
        
        while True:
            if time_limit and time.time() > time_limit:
                break
                
            best_delta = 0
            best_x = -1
            best_y = -1
            
            # Get indices of sets in S
            indices_in_S = np.where(S_vec == 1)[0]
            
            # If S is empty, break (should not happen)
            if len(indices_in_S) == 0:
                break
                
            for x in indices_in_S:
                # Delta(y) = G[y] - G[x] - comp[x][y]
                # We want max Delta(y) for y not in S.
                
                # vector_delta[y] = G[y] - G[x] - comp_matrix[x][y]
                vector_delta = G - G[x] - comp_matrix[x]
                
                # Mask out y in S
                vector_delta[S_vec == 1] = -10000
                
                # Find max
                idx = np.argmax(vector_delta)
                delta_val = vector_delta[idx]
                
                if delta_val > best_delta:
                    best_delta = delta_val
                    best_y = idx
                    best_x = x
                    
            if best_delta > 0:
                # Apply swap
                S_vec[best_x] = 0
                S_vec[best_y] = 1
                
                # Update G
                G += comp_matrix[:, best_y]
                G -= comp_matrix[:, best_x]
                
                current_score += best_delta
                if current_score > best_score:
                    best_score = current_score
                    best_S_vec = S_vec.copy()
            else:
                break
                
        return best_S_vec, best_score
        
    def greedy_init():
        S_vec = np.zeros(total_sets, dtype=np.int32)
        G = np.zeros(total_sets, dtype=np.int32)
        
        # Start with empty set (mask 0)
        S_vec[0] = 1
        G += comp_matrix[:, 0]
        
        for _ in range(m - 1):
            # Find y not in S with max G[y]
            G_masked = G.copy()
            G_masked[S_vec == 1] = -1
            y = np.argmax(G_masked)
            S_vec[y] = 1
            G += comp_matrix[:, y]
            
        return S_vec
        
    best_overall_vec = None
    best_overall_score = -1
    
    # Run ILS
    # First run greedy
    S_vec = greedy_init()
    S_opt, score_opt = local_search_numpy(S_vec, time_limit=deadline - 10)
    best_overall_vec = S_opt
    best_overall_score = score_opt
    
    # ILS loop
    while time.time() < deadline - 5:
        # Perturb
        S_perturbed = best_overall_vec.copy()
        
        # Random swaps
        num_swaps = random.randint(1, 5)
        for _ in range(num_swaps):
            in_S = np.where(S_perturbed == 1)[0]
            if len(in_S) == 0: break
            x = in_S[random.randint(0, len(in_S)-1)]
            
            not_in_S = np.where(S_perturbed == 0)[0]
            if len(not_in_S) == 0: break
            y = not_in_S[random.randint(0, len(not_in_S)-1)]
            
            S_perturbed[x] = 0
            S_perturbed[y] = 1
            
        S_opt, score_opt = local_search_numpy(S_perturbed, time_limit=deadline - 2)
        
        if score_opt > best_overall_score:
            best_overall_score = score_opt
            best_overall_vec = S_opt
            
    # Convert best_overall_vec to lists
    result = []
    indices = np.where(best_overall_vec == 1)[0]
    for mask in indices:
        s = []
        for i in range(n):
            if mask & (1 << i):
                s.append(i)
        result.append(s)
        
    return result
