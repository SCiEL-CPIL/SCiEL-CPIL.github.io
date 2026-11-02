"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Generate all ±1 vectors of weight 1 through 4 in R^11
    # Weight 1: 2*11 = 22
    # Weight 2: 4*C(11,2) = 4*55 = 220
    # Weight 3: 8*C(11,3) = 8*165 = 1320
    # Weight 4: 16*C(11,4) = 16*330 = 5280
    # Total: ~6842 candidates
    
    candidates = []
    
    # Weight 1
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
    
    # Weight 2
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                candidates.append(v)
    
    # Weight 3
    for i, j, k in combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j], v[k] = si, sj, sk
                    candidates.append(v)
    
    # Weight 4
    for i, j, k, l in combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j], v[k], v[l] = si, sj, sk, sl
                        candidates.append(v)
    
    candidates = np.array(candidates)
    norms_sq = np.sum(candidates**2, axis=1)
    n_cand = len(candidates)
    
    # Precompute all pairwise squared distances between candidates
    # This is expensive but we can do it in chunks
    # dist[i,j] = ||v_i - v_j||^2
    
    # Actually, let's use a smarter approach:
    # For vectors with entries in {-1,0,1}, the squared distance between two vectors
    # can be computed from their inner product and norms.
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    
    # Let's precompute the Gram matrix (inner products)
    # But that's 6842^2 which is ~47 million entries, too much memory for int64 but ok for float
    
    # Actually, let's think differently. The key constraint is:
    # For any two selected vectors u, v: ||u-v||^2 >= max(||u||^2, ||v||^2, max_norm_sq)
    # where max_norm_sq is the maximum norm squared among all selected vectors.
    
    # Since we're selecting vectors of various weights, the constraint is:
    # ||u-v||^2 >= M where M = max norm squared in the selected set.
    
    # Strategy: Try different maximum norm values M. For each M, select as many
    # vectors as possible such that all have norm^2 <= M and all pairwise distances >= M.
    
    # Actually, a cleaner approach: 
    # Sort candidates by norm. Try greedy selection with different orderings.
    # But also try: for each possible "max norm" value, do greedy on vectors with norm <= that value,
    # requiring pairwise distance >= that max norm value.
    
    best_selected = None
    best_len = 0
    
    # Try multiple strategies:
    # 1. Greedy by norm ascending
    # 2. Greedy by norm descending  
    # 3. Greedy by random order
    # 4. For each distinct norm value as target max, greedy select vectors with norm <= target
    #    requiring pairwise dist >= target
    
    # First, let's try simple greedy with various orderings
    orderings = [
        np.argsort(norms_sq),      # norm ascending
        np.argsort(-norms_sq),     # norm descending
    ]
    
    rng = np.random.RandomState(42)
    for _ in range(10):
        orderings.append(rng.permutation(n_cand))
    
    for order in orderings:
        if time.time() > deadline:
            break
        
        selected_idx = []
        max_norm_sq = 0
        
        for idx in order:
            if time.time() > deadline:
                break
            
            v = candidates[idx]
            v_norm_sq = int(norms_sq[idx])
            new_max = max(max_norm_sq, v_norm_sq)
            
            if selected_idx:
                selected_vecs = candidates[selected_idx]
                diffs = v - selected_vecs
                dist_sqs = np.sum(diffs**2, axis=1)
                if np.min(dist_sqs) < new_max:
                    continue
            
            selected_idx.append(idx)
            max_norm_sq = new_max
        
        if len(selected_idx) > best_len:
            best_len = len(selected_idx)
            best_selected = selected_idx[:]
    
    # Now try the "target max norm" approach
    # For each distinct norm squared value M, select vectors with norm^2 <= M
    # such that pairwise distance >= M
    
    distinct_norms = np.unique(norms_sq)
    
    for M in distinct_norms:
        if time.time() > deadline:
            break
        
        M_int = int(M)
        
        # Get candidates with norm^2 <= M
        mask = norms_sq <= M
        cand_indices = np.where(mask)[0]
        if len(cand_indices) == 0:
            continue
        
        # Greedy select from these, requiring pairwise dist >= M_int
        # Order by norm descending (heavier vectors first) to maximize packing
        cand_norms = norms_sq[cand_indices]
        order = cand_indices[np.argsort(-cand_norms)]
        
        selected_idx = []
        
        for idx in order:
            if time.time() > deadline:
                break
            
            v = candidates[idx]
            
            if selected_idx:
                selected_vecs = candidates[selected_idx]
                diffs = v - selected_vecs
                dist_sqs = np.sum(diffs**2, axis=1)
                if np.min(dist_sqs) < M_int:
                    continue
            
            selected_idx.append(idx)
        
        if len(selected_idx) > best_len:
            best_len = len(selected_idx)
            best_selected = selected_idx[:]
    
    # Also try: for each M, use random orderings
    for M in distinct_norms:
        if time.time() > deadline:
            break
        
        M_int = int(M)
        mask = norms_sq <= M
        cand_indices = np.where(mask)[0]
        if len(cand_indices) == 0:
            continue
        
        for trial in range(3):
            if time.time() > deadline:
                break
            
            order = cand_indices[rng.permutation(len(cand_indices))]
            
            selected_idx = []
            
            for idx in order:
                if time.time() > deadline:
                    break
                
                v = candidates[idx]
                
                if selected_idx:
                    selected_vecs = candidates[selected_idx]
                    diffs = v - selected_vecs
                    dist_sqs = np.sum(diffs**2, axis=1)
                    if np.min(dist_sqs) < M_int:
                        continue
                
                selected_idx.append(idx)
            
            if len(selected_idx) > best_len:
                best_len = len(selected_idx)
                best_selected = selected_idx[:]
    
    if best_selected is None or best_len == 0:
        # Fallback: weight-2 vectors only
        vectors = []
        for i, j in combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors)
    
    return candidates[best_selected]
# EVOLVE-BLOCK-END
