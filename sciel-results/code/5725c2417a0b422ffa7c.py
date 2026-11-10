import numpy as np
import time
import random
import itertools

def _generate_candidates(d, norm_sq):
    """
    Generate integer vectors in Z^d with exactly the specified norm squared.
    For now, we focus on vectors with entries in {-1, 0, 1, 2}.
    """
    candidates = []
    
    # Case 1: Weight 1 with value +/-2 (Norm sq 4)
    if norm_sq == 4:
        for i in range(d):
            for s in [-2, 2]:
                v = np.zeros(d, dtype=np.int64)
                v[i] = s
                candidates.append(v)
        
        # Case 2: Weight 4 with values +/-1 (Norm sq 4)
        # Indices
        for indices in itertools.combinations(range(d), 4):
            # Signs
            for signs in itertools.product([-1, 1], repeat=4):
                v = np.zeros(d, dtype=np.int64)
                for idx, s in zip(indices, signs):
                    v[idx] = s
                candidates.append(v)
    elif norm_sq == 2:
        for i in range(d):
            for j in range(i + 1, d):
                for si in [-1, 1]:
                    for sj in [-1, 1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        candidates.append(v)
    elif norm_sq == 3:
        for indices in itertools.combinations(range(d), 3):
            for signs in itertools.product([-1, 1], repeat=3):
                v = np.zeros(d, dtype=np.int64)
                for idx, s in zip(indices, signs):
                    v[idx] = s
                candidates.append(v)
                
    return np.array(candidates, dtype=np.int64) if candidates else np.empty((0, d), dtype=np.int64)

def _greedy_pack(candidates, d, max_norm_sq, time_limit):
    """
    Greedy algorithm to find a large subset of candidates such that
    min squared distance >= max_norm_sq.
    """
    if len(candidates) == 0:
        return np.empty((0, d), dtype=np.int64)
        
    best_set = []
    start_time = time.time()
    
    # Try multiple random orderings
    # We want to maximize the set size.
    # Greedy is sensitive to order.
    
    # Optimization: Pre-compute norms? All candidates have same norm sq in this strategy.
    # But we mix types in R=4 case.
    
    # To speed up, we can use a simple check.
    # dist_sq >= max_norm_sq
    
    # We will run a few iterations with different shuffles
    num_restarts = 10
    # Adjust restarts based on time
    # Estimate time per restart: O(K^2 * d) where K is final size.
    # If K ~ 500, 250k * 11 = 2.7M ops. ~0.1s in Python? Maybe 0.5s.
    # 10 restarts ~ 5s. Safe.
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit * 0.8:
            break
            
        indices = np.arange(len(candidates))
        np.random.shuffle(indices)
        shuffled_cands = candidates[indices]
        
        current_set = []
        
        for v in shuffled_cands:
            # Check against all in current_set
            valid = True
            for u in current_set:
                diff = v - u
                dist_sq = np.sum(diff * diff)
                if dist_sq < max_norm_sq:
                    valid = False
                    break
            
            if valid:
                current_set.append(v)
        
        if len(current_set) > len(best_set):
            best_set = current_set
            
        # Early termination if we found a perfect packing? Unlikely.
        
    return np.array(best_set, dtype=np.int64) if best_set else np.empty((0, d), dtype=np.int64)

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 600 # Safety margin
    
    # Strategy 1: R=2 (Known 220)
    # We can generate this instantly.
    cands_r2 = _generate_candidates(d, 2)
    # For R=2, all are compatible.
    set_r2 = cands_r2.copy()
    best_k = len(set_r2)
    best_arr = set_r2
    
    # Strategy 2: R=4
    # Candidates: 22 (weight 1, val 2) + 5280 (weight 4, val 1)
    cands_r4 = _generate_candidates(d, 4)
    
    # We must ensure the max norm sq in the set is 4.
    # All candidates have norm sq 4.
    # So max_norm_sq = 4.
    
    # Greedy pack
    # We want to be fast.
    # Let's run greedy for a few seconds.
    t0 = time.time()
    # Allocate 60% of remaining time to R=4 search
    time_alloc_r4 = min(300, (deadline - time.time()) * 0.6)
    
    set_r4 = _greedy_pack(cands_r4, d, 4, time_alloc_r4)
    
    if len(set_r4) > best_k:
        best_k = len(set_r4)
        best_arr = set_r4
        
    # Strategy 3: R=3
    # Candidates: 1320 (weight 3, val 1)
    # Max norm sq = 3.
    # Min dist sq >= 3.
    
    t1 = time.time()
    time_alloc_r3 = min(150, (deadline - time.time()) * 0.3)
    
    if time_alloc_r3 > 10: # Only if we have time
        cands_r3 = _generate_candidates(d, 3)
        set_r3 = _greedy_pack(cands_r3, d, 3, time_alloc_r3)
        
        if len(set_r3) > best_k:
            best_k = len(set_r3)
            best_arr = set_r3

    # Final verification and formatting
    if len(best_arr) == 0:
        # Fallback to a known valid small set
        return np.array([[1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        
    # Check validity
    k = best_arr.shape[0]
    norms = np.sum(best_arr**2, axis=1)
    max_norm = np.max(norms)
    
    # Pairwise check
    valid = True
    for i in range(k):
        for j in range(i+1, k):
            diff = best_arr[i] - best_arr[j]
            dist_sq = np.sum(diff**2)
            if dist_sq < max_norm:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # If our best candidate is invalid, fall back to R=2 set
        # which we know is valid
        return cands_r2

    # Round and cast
    result = np.around(best_arr).astype(np.int64)
    return result
