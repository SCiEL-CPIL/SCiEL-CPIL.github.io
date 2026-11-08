import time
import numpy as np
import itertools
import random

D = 11

def construct():
    deadline = time.time() + 140
    
    # Generate candidate pool: all integer vectors with small norm
    # Norm 1: 22 vectors
    # Norm sqrt(2): 220 vectors  
    # Norm sqrt(5): vectors with one ±2 and one ±1, or three non-zero entries
    # Let's include norm sqrt(5) vectors: C(11,1)*C(10,1)*4 (one 2, one 1) + C(11,3)*8 (three ±1)
    
    candidates = []
    
    # Norm 1
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
    
    # Norm sqrt(2)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
    
    # Norm sqrt(5): type 1: one entry ±2, one entry ±1
    for i, j in itertools.permutations(range(D), 2):
        for s2 in (1, -1):
            for s1 in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = 2 * s2
                v[j] = s1
                candidates.append(v)
    
    # Norm sqrt(5): type 2: three entries ±1
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Compute norms squared
    norms_sq = np.sum(candidates**2, axis=1)
    max_norm_sq_global = np.max(norms_sq)
    
    # For a set S, the constraint is: min_{u,v in S, u≠v} ||u-v||^2 >= max_{w in S} ||w||^2
    # This is tricky because max norm depends on the set.
    
    # Strategy: fix a target max norm squared T. Include only candidates with norm^2 <= T.
    # Then require pairwise distances >= T.
    # Try different T values and find the best.
    
    best_result = None
    best_k = 0
    
    # Try T = 1, 2, 5
    for T in [1, 2, 5]:
        if time.time() > deadline - 10:
            break
        
        # Filter candidates with norm^2 <= T
        mask = norms_sq <= T
        cand_subset = candidates[mask]
        k_sub = len(cand_subset)
        
        if k_sub == 0:
            continue
        
        # Build compatibility: u,v compatible if ||u-v||^2 >= T
        # Greedy with multiple orderings
        
        best_for_T = None
        best_k_for_T = 0
        
        # Try several random orderings
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            
            order = np.random.permutation(k_sub)
            selected = []
            selected_set = set()
            
            for idx in order:
                if time.time() > deadline - 10:
                    break
                v = cand_subset[idx]
                compatible = True
                for u in selected:
                    dist_sq = np.sum((u - v)**2)
                    if dist_sq < T:
                        compatible = False
                        break
                if compatible:
                    selected.append(v)
            
            k_sel = len(selected)
            if k_sel > best_k_for_T:
                best_k_for_T = k_sel
                best_for_T = np.array(selected, dtype=np.int64)
        
        if best_for_T is not None and best_k_for_T > best_k:
            best_k = best_k_for_T
            best_result = best_for_T
    
    if best_result is None:
        # Fallback: just norm-sqrt(2) vectors
        result = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    result.append(v)
        best_result = np.array(result, dtype=np.int64)
    
    return best_result
