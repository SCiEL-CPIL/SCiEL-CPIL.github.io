import time
import numpy as np
from itertools import product

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Generate candidate vectors of various weights
    candidates = []
    
    # Weight-1 vectors: one non-zero entry of ±1
    for i in range(D):
        for sign in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = sign
            candidates.append(v)
    
    # Weight-2 vectors: two non-zero entries of ±1
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j] = si, sj
                    candidates.append(v)
    
    # Weight-3 vectors: three non-zero entries of ±1
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            v = np.zeros(D, dtype=np.int64)
                            v[i], v[j], v[k] = si, sj, sk
                            candidates.append(v)
    
    # Weight-4 vectors: four non-zero entries of ±1
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for l in range(k+1, D):
                    for si in (1, -1):
                        for sj in (1, -1):
                            for sk in (1, -1):
                                for sl in (1, -1):
                                    v = np.zeros(D, dtype=np.int64)
                                    v[i], v[j], v[k], v[l] = si, sj, sk, sl
                                    candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    
    # Compute squared norms
    norms_sq = np.sum(candidates ** 2, axis=1)
    
    # Greedy farthest-point sampling
    # We want to select vectors such that min pairwise dist >= max norm
    # Strategy: start with a vector, then iteratively add the candidate that
    # maximizes the minimum distance to the current set, subject to the constraint
    
    selected = []
    selected_norms_sq = []
    
    # Start with a weight-4 vector to maximize the threshold
    # Actually, let's try starting with different strategies
    
    # Strategy: Use farthest-point sampling with constraint checking
    # For each candidate, check if adding it maintains the constraint
    
    def can_add(candidate, selected_arr, current_max_norm_sq):
        """Check if candidate can be added to selected set"""
        if len(selected_arr) == 0:
            return True
        # Compute distances to all selected vectors
        diffs = candidate[None, :] - selected_arr
        dists_sq = np.sum(diffs ** 2, axis=1)
        min_dist_sq = np.min(dists_sq)
        # The new max norm will be max(current_max, candidate_norm_sq)
        new_max_norm_sq = max(current_max_norm_sq, np.sum(candidate ** 2))
        # Constraint: min_dist_sq >= new_max_norm_sq
        return min_dist_sq >= new_max_norm_sq
    
    # Try multiple starting points and keep the best
    best_selected = []
    best_count = 0
    
    # First, let's try a simple greedy approach starting from a high-norm vector
    # Find weight-4 vectors (norm_sq = 4)
    w4_indices = np.where(norms_sq == 4)[0]
    
    if len(w4_indices) > 0:
        # Start with the first weight-4 vector
        start_idx = w4_indices[0]
        selected = [candidates[start_idx]]
        selected_arr = np.array(selected)
        current_max_norm_sq = norms_sq[start_idx]
        
        # Iteratively add the candidate that maximizes min distance
        remaining = list(range(len(candidates)))
        remaining.remove(start_idx)
        
        while remaining:
            if time.time() > deadline:
                break
            
            best_idx = -1
            best_min_dist = -1
            
            for idx in remaining:
                cand = candidates[idx]
                if not can_add(cand, selected_arr, current_max_norm_sq):
                    continue
                
                # Compute min distance to selected set
                diffs = cand[None, :] - selected_arr
                dists_sq = np.sum(diffs ** 2, axis=1)
                min_dist = np.min(dists_sq)
                
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = idx
            
            if best_idx == -1:
                break
            
            selected.append(candidates[best_idx])
            selected_arr = np.array(selected)
            current_max_norm_sq = max(current_max_norm_sq, norms_sq[best_idx])
            remaining.remove(best_idx)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_selected = selected[:]
    
    # Also try starting with weight-2 vectors (the original approach)
    w2_indices = np.where(norms_sq == 2)[0]
    
    if len(w2_indices) > 0:
        # Try a few different starting points
        for trial_start in w2_indices[:10]:
            if time.time() > deadline:
                break
            
            selected = [candidates[trial_start]]
            selected_arr = np.array(selected)
            current_max_norm_sq = norms_sq[trial_start]
            
            remaining = list(range(len(candidates)))
            remaining.remove(trial_start)
            
            while remaining:
                if time.time() > deadline:
                    break
                
                best_idx = -1
                best_min_dist = -1
                
                for idx in remaining:
                    cand = candidates[idx]
                    if not can_add(cand, selected_arr, current_max_norm_sq):
                        continue
                    
                    diffs = cand[None, :] - selected_arr
                    dists_sq = np.sum(diffs ** 2, axis=1)
                    min_dist = np.min(dists_sq)
                    
                    if min_dist > best_min_dist:
                        best_min_dist = min_dist
                        best_idx = idx
                
                if best_idx == -1:
                    break
                
                selected.append(candidates[best_idx])
                selected_arr = np.array(selected)
                current_max_norm_sq = max(current_max_norm_sq, norms_sq[best_idx])
                remaining.remove(best_idx)
            
            if len(selected) > best_count:
                best_count = len(selected)
                best_selected = selected[:]
    
    # Also try the original approach as a baseline
    original_vectors = []
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j] = si, sj
                    original_vectors.append(v)
    
    original_arr = np.array(original_vectors, dtype=np.int64)
    
    if len(original_arr) > best_count:
        best_selected = original_vectors[:]
        best_count = len(original_arr)
    
    return np.array(best_selected, dtype=np.int64)
