import numpy as np
import time
import random

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 585  # Safety margin for 600s limit
    
    # The 220 vectors with norm^2 = 2 (exactly two ±1 entries) form a valid set:
    # For any two distinct such vectors u, v: u·v ∈ {-2,-1,0,1,2}.
    # ||u-v||^2 = 4 - 2(u·v) >= 4-2 = 2 = R.
    # So the base set of all 220 norm-2 vectors is valid with k=220.
    
    # Strategy: Start with the 220 norm-2 vectors as the base.
    # Try to add additional vectors (norm^2 = 3, 4, 5, ...) one by one,
    # checking if they are at distance >= new_R from all existing points.
    # When adding a vector with norm^2 > current R, the new R becomes that norm^2.
    # We must verify all existing pairs still satisfy dist >= new_R.
    
    # Actually, a better approach:
    # The constraint is: min_pairwise_dist^2 >= max_norm^2.
    # If we have a set S with max_norm^2 = R, all pairs must have dist^2 >= R.
    # The 220 norm-2 vectors satisfy this with R=2.
    # Can we add more?
    # If we add a vector v with norm^2 = 3, then R becomes 3.
    # Now ALL pairs must have dist^2 >= 3.
    # Among the 220 norm-2 vectors, the minimum pairwise dist^2 is 2 (when they share one coordinate).
    # So adding any norm-3 vector would make R=3, but the existing pairs with dist^2=2 would violate the constraint.
    # Therefore, we CANNOT add norm-3 vectors to the full set of 220.
    
    # What if we REMOVE some norm-2 vectors to make room for norm-3 vectors?
    # If we remove vectors that are at distance 2 from each other, we might be able to add norm-3 vectors.
    # But this is a complex optimization.
    
    # Alternative: Can we find a LARGER set where all vectors have norm^2 = 2?
    # The 220 vectors are ALL vectors with norm^2 = 2 in 11D. So no.
    
    # What about a different R?
    # R=1: Only norm-1 vectors. 22 vectors. Max k=22 (all of them, since min dist^2 = 2 >= 1).
    # R=2: 220 vectors. k=220.
    # R=3: Need all pairs dist^2 >= 3. 
    #   Norm-1 vectors: 22. Pairwise dist^2 = 2 < 3. So can't use all 22.
    #   Norm-2 vectors: 220. Pairwise min dist^2 = 2 < 3. Can't use all 220.
    #   Norm-3 vectors: 1320. 
    #   We need a subset where all pairwise dist^2 >= 3.
    #   This is a packing problem.
    
    # Let's focus on R=2 since it gives 220, and see if we can do better with R=3 or R=4.
    
    # For R=2, the answer is exactly 220 (all norm-2 vectors).
    # Let's verify this is valid and return it.
    
    # Generate all 220 norm-2 vectors
    base_vectors = []
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    base_vectors.append(v)
    
    base_array = np.array(base_vectors, dtype=np.int64)
    k_base = len(base_array)  # Should be 220
    
    # Verify base set
    max_norm_base = np.max(np.sum(base_array**2, axis=1))  # Should be 2
    valid_base = True
    for i in range(k_base):
        for j in range(i + 1, k_base):
            d2 = np.dot(base_array[i] - base_array[j], base_array[i] - base_array[j])
            if d2 < max_norm_base:
                valid_base = False
                break
        if not valid_base:
            break
    
    best_set = base_array
    best_k = k_base if valid_base else 0
    
    # Now try R=3: Find max subset of vectors with norm^2 <= 3 such that all pairwise dist^2 >= 3.
    # Candidates: norm 1 (22), norm 2 (220), norm 3 (1320). Total 1562.
    # We need a large independent set in the conflict graph where conflict = dist^2 < 3.
    
    # Generate all candidates with norm^2 <= 3
    candidates_3 = []
    # Norm 1
    for i in range(d):
        v = np.zeros(d, dtype=np.int64)
        v[i] = 1
        candidates_3.append(v)
        candidates_3.append(-v)
    # Norm 2
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    candidates_3.append(v)
    # Norm 3
    for i in range(d):
        for j in range(i + 1, d):
            for k in range(j + 1, d):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        for s3 in [1, -1]:
                            v = np.zeros(d, dtype=np.int64)
                            v[i] = s1
                            v[j] = s2
                            v[k] = s3
                            candidates_3.append(v)
    
    cand3 = np.array(candidates_3, dtype=np.int64)
    n3 = len(cand3)
    norms3 = np.sum(cand3**2, axis=1)
    
    # Compute distance matrix
    dots3 = cand3 @ cand3.T
    dist3 = norms3[:, None] + norms3[None, :] - 2 * dots3
    
    # Conflict if dist < 3
    conflict3 = dist3 < 3
    np.fill_diagonal(conflict3, False)
    deg3 = conflict3.sum(axis=1)
    
    # Greedy MIS with multiple trials
    for trial in range(30):
        if time.time() > deadline:
            break
            
        available = np.ones(n3, dtype=bool)
        selected = []
        
        while available.any():
            avail_idx = np.where(available)[0]
            if len(avail_idx) == 0:
                break
                
            avail_degrees = deg3[avail_idx]
            min_deg = np.min(avail_degrees)
            min_nodes = avail_idx[avail_degrees == min_deg]
            
            if trial == 0:
                pick_idx = min_nodes[0]
            else:
                pick_idx = min_nodes[random.randint(0, len(min_nodes) - 1)]
                
            selected.append(pick_idx)
            
            to_remove = conflict3[pick_idx] | (np.arange(n3) == pick_idx)
            available[to_remove] = False
            
        current_size = len(selected)
        if current_size > best_k:
            best_k = current_size
            best_set = cand3[selected]
    
    # Try R=4
    # Candidates: norm <= 4. Total ~ 1562 + 5302 = 6864.
    # This might be too slow for greedy with 30 trials on 6864 nodes.
    # Let's try a few trials.
    
    candidates_4 = list(candidates_3)
    # Norm 4 type 1: (2, 0, ...)
    for i in range(d):
        v = np.zeros(d, dtype=np.int64)
        v[i] = 2
        candidates_4.append(v)
        candidates_4.append(-v)
    # Norm 4 type 2: (1,1,1,1,0,...)
    for i in range(d):
        for j in range(i+1, d):
            for k in range(j+1, d):
                for l in range(k+1, d):
                    for s1 in [1, -1]:
                        for s2 in [1, -1]:
                            for s3 in [1, -1]:
                                for s4 in [1, -1]:
                                    v = np.zeros(d, dtype=np.int64)
                                    v[i] = s1
                                    v[j] = s2
                                    v[k] = s3
                                    v[l] = s4
                                    candidates_4.append(v)
    
    cand4 = np.array(candidates_4, dtype=np.int64)
    n4 = len(cand4)
    norms4 = np.sum(cand4**2, axis=1)
    
    # Only proceed if n4 is manageable
    if n4 <= 8000:
        dots4 = cand4 @ cand4.T
        dist4 = norms4[:, None] + norms4[None, :] - 2 * dots4
        conflict4 = dist4 < 4
        np.fill_diagonal(conflict4, False)
        deg4 = conflict4.sum(axis=1)
        
        for trial in range(10):
            if time.time() > deadline:
                break
                
            available = np.ones(n4, dtype=bool)
            selected = []
            
            while available.any():
                avail_idx = np.where(available)[0]
                if len(avail_idx) == 0:
                    break
                    
                avail_degrees = deg4[avail_idx]
                min_deg = np.min(avail_degrees)
                min_nodes = avail_idx[avail_degrees == min_deg]
                
                if trial == 0:
                    pick_idx = min_nodes[0]
                else:
                    pick_idx = min_nodes[random.randint(0, len(min_nodes) - 1)]
                    
                selected.append(pick_idx)
                
                to_remove = conflict4[pick_idx] | (np.arange(n4) == pick_idx)
                available[to_remove] = False
                
            current_size = len(selected)
            if current_size > best_k:
                best_k = current_size
                best_set = cand4[selected]
    
    # Final verification
    if best_k >= 2:
        result = np.array(best_set, dtype=np.int64)
        max_norm = np.max(np.sum(result**2, axis=1))
        valid = True
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                d2 = np.dot(result[i] - result[j], result[i] - result[j])
                if d2 < max_norm:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return result
    
    # Fallback: return the base 220 set if valid, else a minimal valid set
    if valid_base:
        return base_array
    else:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
