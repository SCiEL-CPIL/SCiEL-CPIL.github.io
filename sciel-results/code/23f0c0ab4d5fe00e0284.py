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
            # Check even sum
            if sum(signs) % 2 == 0:
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
    degrees = conflicts.sum(axis=1).astype(np.int32)
    
    best_set = np.zeros(N, dtype=bool)
    best_count = 0
    
    # Use more restarts and a better local search
    n_restarts = 500
    
    for r in range(n_restarts):
        if time.time() - start_time > 100:
            break
            
        # Greedy phase
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        # Copy degrees to avoid modifying original
        cur_degrees = degrees.copy()
        
        while np.any(alive):
            alive_idx = np.where(alive)[0]
            if len(alive_idx) == 0: break
            
            min_deg = np.min(cur_degrees[alive_idx])
            cands = alive_idx[cur_degrees[alive_idx] == min_deg]
            chosen = cands[np.random.randint(len(cands))]
            
            in_set[chosen] = True
            
            nbrs = neighbors[chosen]
            to_remove = np.union1d([chosen], nbrs)
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                cur_degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        count = np.sum(in_set)
        
        # Local Search
        improved = True
        swap_limit = 500 # Limit swaps per restart to prevent infinite loops or timeouts
        swaps_done = 0
        while improved and swaps_done < swap_limit:
            improved = False
            in_set_indices = np.where(in_set)[0]
            if len(in_set_indices) == 0: break
            np.random.shuffle(in_set_indices)
            
            for u in in_set_indices:
                if time.time() - start_time > 110:
                    break
                
                nbrs_u = neighbors[u]
                candidates = []
                for x in nbrs_u:
                    if in_set[x]:
                        continue
                    if np.sum(conflicts[x, in_set]) == 1:
                        candidates.append(x)
                        
                if len(candidates) >= 2:
                    found_pair = False
                    # Sort candidates by degree to pick "less conflicting" ones?
                    # Or just random
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
                                swaps_done += 1
                                break
                                
            if time.time() - start_time > 110:
                break

        if count > best_count:
            best_count = count
            best_set = in_set.copy()
            
    V_t2_sel = V_t2[best_set]
    result = np.vstack([V_t1, V_t2_sel])
    
    return result
