import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    
    # 1. Generate D11 Minimal Vectors
    # Type 1: Weight 1, +/- 2
    vecs_t1 = []
    for i in range(D):
        for s in [1, -1]:
            v = [0]*D
            v[i] = 2*s
            vecs_t1.append(v)
            
    # Type 2: Weight 4, +/- 1, even sum
    vecs_t2 = []
    for pos in itertools.combinations(range(D), 4):
        for mask in range(16):
            signs = [1 if (mask & (1<<j)) == 0 else -1 for j in range(4)]
            v = [0]*D
            for p, val in zip(pos, signs):
                v[p] = val
            vecs_t2.append(v)
                
    V_t1 = np.array(vecs_t1, dtype=np.int64)
    V_t2 = np.array(vecs_t2, dtype=np.int64)
    N = len(V_t2)
    
    # 2. Compute Conflict Matrix for Type 2
    # Conflict if dot product > 2
    dots = V_t2 @ V_t2.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    
    best_set = np.zeros(N, dtype=bool)
    best_count = 0
    
    # Increase restarts, use time limit
    n_restarts = 1000
    
    for r in range(n_restarts):
        if time.time() - start_time > 100:
            break
            
        # Greedy Phase: Randomized Min-Degree
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = conflicts.sum(axis=1).astype(np.int32)
        
        while np.any(alive):
            alive_idx = np.where(alive)[0]
            if len(alive_idx) == 0: break
            
            min_deg = np.min(degrees[alive_idx])
            cands = alive_idx[degrees[alive_idx] == min_deg]
            chosen = cands[np.random.randint(len(cands))]
            
            in_set[chosen] = True
            
            nbrs = neighbors[chosen]
            to_remove = np.union1d([chosen], nbrs)
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        count = np.sum(in_set)
        
        # Local Search: 1-for-2 swap
        # We try to improve the set by removing one vertex and adding two non-conflicting ones.
        # To speed up, we limit the number of swap attempts per restart.
        improved = True
        swap_attempts = 0
        max_swaps_per_restart = 50
        
        while improved and swap_attempts < max_swaps_per_restart:
            improved = False
            swap_attempts += 1
            in_set_indices = np.where(in_set)[0]
            np.random.shuffle(in_set_indices)
            
            for u in in_set_indices:
                if time.time() - start_time > 110:
                    break
                if improved:
                    break # Restart the while loop to refresh indices? Or just continue?
                    # Actually, if we improved, the set changed. We should probably restart the scan.
                    # But for speed, we can just break and let the while loop handle it.
                    
                nbrs_u = neighbors[u]
                
                candidates = []
                # Check neighbors of u
                for x in nbrs_u:
                    if in_set[x]:
                        continue
                    # Check if x has exactly 1 neighbor in in_set (which is u)
                    # np.sum(conflicts[x, in_set]) is O(N)
                    # Optimization: If we maintain a count, it's O(1).
                    # For now, we accept the cost but limit candidates.
                    if np.sum(conflicts[x, in_set]) == 1:
                        candidates.append(x)
                        
                if len(candidates) >= 2:
                    # Try to find a compatible pair in candidates
                    found_pair = False
                    for i in range(len(candidates)):
                        if found_pair: break
                        for j in range(i+1, len(candidates)):
                            if not conflicts[candidates[i], candidates[j]]:
                                in_set[u] = False
                                in_set[candidates[i]] = True
                                in_set[candidates[j]] = True
                                count += 1
                                improved = True
                                found_pair = True
                                break
                                
        if count > best_count:
            best_count = count
            best_set = in_set.copy()
            
    # Combine results
    V_t2_sel = V_t2[best_set]
    result = np.vstack([V_t1, V_t2_sel])
    
    return result
