import random
import time
import numpy as np
from itertools import combinations

def construct():
    # Generate all 4-subsets of [1, 16] as bitmasks
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        all_sets.append(mask)
    
    n_sets = len(all_sets)
    all_sets_np = np.array(all_sets, dtype=np.int32)
    
    best_family = []
    best_size = 0
    
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin for conversion and safety
    
    for seed in range(10000):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        indices = np.arange(n_sets)
        np.random.shuffle(indices)
        
        family = []
        family_np = np.array([], dtype=np.int32)
        
        for idx in indices:
            s = all_sets_np[idx]
            
            if len(family_np) == 0:
                family.append(s)
                family_np = np.array([s], dtype=np.int32)
                continue
                
            # Check conflicts
            # fams is shape (N,)
            # inter_ab is shape (N, N)
            inter_ab = family_np[:, None] & family_np[None, :]
            inter_as = family_np & s
            
            # We need to check if there exists i != j such that:
            # inter_ab[i, j] == inter_as[i] and inter_ab[i, j] == inter_as[j]
            # Note: inter_ab[i, j] == inter_ab[j, i]
            # So we can just check (inter_ab == inter_as[:, None]) & (inter_ab == inter_as[None, :])
            
            # To avoid checking i == j (which is not a pair of distinct sets in family),
            # we can set diagonal to False or just rely on the fact that 
            # if A_i == A_j, they are the same set, but family has distinct sets.
            # However, the condition "three distinct sets" implies A != B.
            # In our family, all sets are distinct.
            # So i != j is required.
            
            conflict_matrix = (inter_ab == inter_as[:, None]) & (inter_ab == inter_as[None, :])
            
            # Exclude diagonal
            np.fill_diagonal(conflict_matrix, False)
            
            if conflict_matrix.any():
                continue
            else:
                family.append(s)
                family_np = np.append(family_np, s)
                
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            
    # Convert best_family bitmasks back to sorted lists of 1-based integers
    result = []
    for mask in best_family:
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(elems)
        
    return result
