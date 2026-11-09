import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 16
    k = 4
    max_size = 250
    
    all_sets = [set(c) for c in combinations(range(1, n + 1), k)]
    total = len(all_sets)
    set_to_idx = {s: i for i, s in enumerate(all_sets)}
    
    # Precompute forbidden lists as numpy arrays for faster updating
    forbidden = [[None] * total for _ in range(total)]
    
    for i in range(total):
        A = all_sets[i]
        for j in range(i + 1, total):
            B = all_sets[j]
            inter = A & B
            union = A | B
            R = set(range(1, n + 1)) - union
            size_needed = k - len(inter)
            
            forb_list = []
            if size_needed <= len(R):
                for S in combinations(R, size_needed):
                    C = inter.union(S)
                    c_idx = set_to_idx.get(C)
                    if c_idx is not None:
                        forb_list.append(c_idx)
            
            # Convert to numpy array
            forb_arr = np.array(forb_list, dtype=np.int32)
            forbidden[i][j] = forb_arr
            forbidden[j][i] = forb_arr
            
            if time.time() > deadline - 15.0:
                break
        if time.time() > deadline - 15.0:
            break
            
    def greedy_build(seed):
        random.seed(seed)
        order = list(range(total))
        random.shuffle(order)
        
        family = []
        family_set = set()
        conflict_count = np.zeros(total, dtype=np.int32)
        
        for idx in order:
            if len(family) >= max_size:
                break
            if conflict_count[idx] > 0:
                continue
                
            family.append(idx)
            family_set.add(idx)
            
            for f_idx in family:
                if f_idx == idx:
                    continue
                # Update conflict counts
                if forbidden[idx][f_idx] is not None:
                    conflict_count[forbidden[idx][f_idx]] += 1
                else:
                    # Fallback if precomputation was cut short
                    A = all_sets[idx]
                    B = all_sets[f_idx]
                    inter = A & B
                    union = A | B
                    R = set(range(1, n + 1)) - union
                    size_needed = k - len(inter)
                    if size_needed <= len(R):
                        for S in combinations(R, size_needed):
                            C = inter.union(S)
                            c_idx = set_to_idx.get(C)
                            if c_idx is not None:
                                conflict_count[c_idx] += 1
                                
        return family

    best_family = []
    best_size = 0
    attempts = 0
    max_attempts = 2000
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 5.0:
            break
        family = greedy_build(attempt)
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]

    return [sorted(all_sets[idx]) for idx in best_family]

