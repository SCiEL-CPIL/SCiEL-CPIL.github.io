import numpy as np
from itertools import combinations
import random

def construct():
    D = 11
    all_subsets = list(combinations(range(D), 4))
    n_subsets = len(all_subsets)
    
    # Precompute conflicts
    conflicts = [[] for _ in range(n_subsets)]
    for i in range(n_subsets):
        for j in range(i + 1, n_subsets):
            if len(set(all_subsets[i]) & set(all_subsets[j])) >= 3:
                conflicts[i].append(j)
                conflicts[j].append(i)
                
    best_count = 0
    best_selected = []
    
    # Try multiple random orderings or strategies
    for _ in range(100):
        # Strategy: Greedy with "Least Future Conflicts"
        available = set(range(n_subsets))
        selected = []
        
        while available:
            # Find subset in available with min conflicts within available
            min_conf = n_subsets + 1
            best_s = -1
            candidates = []
            
            for s in available:
                conf_count = 0
                for c in conflicts[s]:
                    if c in available:
                        conf_count += 1
                if conf_count < min_conf:
                    min_conf = conf_count
                    candidates = [s]
                elif conf_count == min_conf:
                    candidates.append(s)
            
            if not candidates:
                break
                
            # Randomly pick from candidates to break ties
            best_s = random.choice(candidates)
            
            selected.append(best_s)
            available.remove(best_s)
            # Remove all conflicts of best_s from available
            for c in conflicts[best_s]:
                if c in available:
                    available.remove(c)
                    
        if len(selected) > best_count:
            best_count = len(selected)
            best_selected = selected
            
    # Generate vectors
    vectors = []
    for idx in best_selected:
        subset = all_subsets[idx]
        i, j, k, l = subset
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = [0] * D
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        vectors.append(v)
                        
    return np.array(vectors, dtype=np.int64)
