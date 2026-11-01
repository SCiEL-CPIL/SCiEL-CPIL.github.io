import numpy as np
import time
import random
from itertools import combinations, product

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    best_vectors = None
    best_count = 0
    
    # Generate candidate pool: integer vectors with small entries
    # Focus on vectors with entries in {-1, 0, 1} and Hamming weight 2 or 3
    candidates = []
    
    # Weight 2: norm^2 = 2
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0]*D
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    # Weight 3: norm^2 = 3
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            v = [0]*D
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cands = len(candidates)
    
    # For each candidate, compute norm^2
    norms_sq = np.sum(candidates**2, axis=1)
    
    def is_valid_set(vectors, max_norm_sq):
        """Check if set of vectors satisfies min pairwise dist >= max_norm."""
        n = len(vectors)
        for i in range(n):
            for j in range(i+1, n):
                diff = vectors[i] - vectors[j]
                dist_sq = int(np.dot(diff, diff))
                if dist_sq < max_norm_sq:
                    return False
        return True
    
    def greedy_with_norm(candidates, norms_sq, order, target_norm_sq):
        """Greedy: only consider candidates with norm^2 == target_norm_sq."""
        selected = []
        for idx in order:
            if time.time() > deadline:
                break
            if norms_sq[idx] != target_norm_sq:
                continue
            v = candidates[idx]
            
            if not selected:
                selected.append(v)
                continue
            
            # Check against all selected
            ok = True
            for sv in selected:
                diff = v - sv
                dist_sq = int(np.dot(diff, diff))
                if dist_sq < target_norm_sq:
                    ok = False
                    break
            
            if ok:
                selected.append(v)
        
        return selected
    
    # Try different target norms
    for target_norm_sq in [2, 3]:
        if time.time() > deadline:
            break
        
        # Filter candidates to those with this norm
        mask = norms_sq == target_norm_sq
        filtered_indices = np.where(mask)[0]
        
        if len(filtered_indices) == 0:
            continue
        
        n_filtered = len(filtered_indices)
        
        # Try multiple random orderings
        n_tries = 200
        for trial in range(n_tries):
            if time.time() > deadline:
                break
            
            random.seed(trial * 777 + 42)
            order = list(filtered_indices)
            random.shuffle(order)
            
            result = greedy_with_norm(candidates, norms_sq, order, target_norm_sq)
            if len(result) > best_count:
                best_count = len(result)
                best_vectors = result
    
    if best_vectors is None:
        # Fallback
        vectors = []
        for i in range(D):
            for j in range(i+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = [0]*D
                        v[i] = si
                        v[j] = sj
                        vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return np.array(best_vectors, dtype=np.int64)
