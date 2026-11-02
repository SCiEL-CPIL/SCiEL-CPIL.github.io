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
        # Generate sign patterns with even number of -1s
        # There are 8 such patterns for 4 positions
        for mask in range(16):
            # Count bits
            if bin(mask).count('1') % 2 == 0:
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
    
    # 3. Find Independent Set using Greedy + Local Search (Swap)
    best_set = np.zeros(N, dtype=bool)
    best_count = 0
    
    # We will run multiple rounds of Greedy + Local Search
    while time.time() - start_time < 110:
        # Greedy Phase
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = conflicts.sum(axis=1).astype(np.int32)
        
        # Randomized Greedy
        while np.any(alive):
            alive_idx = np.where(alive)[0]
            if len(alive_idx) == 0: break
            
            # Pick random from min-degree candidates
            min_deg = np.min(degrees[alive_idx])
            cands = alive_idx[degrees[alive_idx] == min_deg]
            chosen = cands[np.random.randint(len(cands))]
            
            in_set[chosen] = True
            
            nbrs = np.where(conflicts[chosen])[0]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        count = np.sum(in_set)
        
        # Local Search Phase
        # Try to improve by swapping
        improved = True
        while improved and time.time() - start_time < 110:
            improved = False
            # Try random swaps
            for _ in range(50): # Limit swaps per round to avoid stalling
                if time.time() - start_time > 110:
                    break
                if not np.any(in_set):
                    break
                    
                # Pick a random vertex in set
                set_indices = np.where(in_set)[0]
                u = set_indices[np.random.randint(len(set_indices))]
                
                # Remove u
                in_set[u] = False
                current_count = count - 1
                
                # Find all vertices not in set that can now be added
                # A vertex v can be added if it has no neighbors in the current set
                # Neighbors in set: conflicts[v] & in_set
                # We want sum(conflicts[v] & in_set) == 0
                
                # Efficient check:
                # For each v not in set, check if any neighbor is in set.
                # This is expensive if done naively for all v.
                # But we can just try to add candidates.
                
                # Candidates are neighbors of u that were removed?
                # No, any vertex that was blocked by u might be free now.
                # Actually, any vertex v such that (conflicts[v] & in_set).sum() == 0.
                
                # Let's just iterate through all vertices not in set?
                # N is 2640. 2640 * 50 swaps * restarts.
                # 2640 * 50 = 132k checks.
                # Each check is a bit operation or sum.
                # Should be fast enough.
                
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    in_set[u] = True # Revert
                    count = current_count + 1
                    break
                    
                # Check which not_in_set vertices are now free
                # conflicts[not_in_set] @ in_set? No, conflicts is boolean.
                # We need (conflicts[not_in_set] & in_set).sum(axis=1) == 0
                
                # This creates a large temp array.
                # Better:
                # For each v in not_in_set, check if any neighbor is in set.
                
                # Optimization:
                # Only check vertices that were neighbors of u?
                # No, removing u might free up a vertex that wasn't a neighbor of u?
                # If v was not a neighbor of u, its set of neighbors in `in_set` didn't change (since u wasn't in it).
                # So v was already blocked by some other neighbor.
                # So only neighbors of u can become free.
                
                u_nbrs = np.where(conflicts[u])[0]
                candidates = u_nbrs[~in_set[u_nbrs]]
                
                if len(candidates) == 0:
                    in_set[u] = True # Revert
                    count = current_count + 1
                    continue
                    
                # Check which candidates are now free
                # A candidate v is free if it has no neighbors in the *current* in_set (which excludes u).
                
                free_cands = []
                for v in candidates:
                    # Check if v has any neighbor in in_set
                    # in_set is boolean array
                    # conflicts[v] is boolean array
                    # (conflicts[v] & in_set).sum() == 0
                    if np.sum(conflicts[v] & in_set) == 0:
                        free_cands.append(v)
                        
                if len(free_cands) > 0:
                    # Add all free candidates
                    for v in free_cands:
                        in_set[v] = True
                    new_count = count + len(free_cands)
                    
                    if new_count > count:
                        count = new_count
                        improved = True
                    else:
                        # Revert
                        in_set[u] = True
                        for v in free_cands:
                            in_set[v] = False
                        # count remains same (was decremented, now we didn't add enough to offset? 
                        # Actually if new_count <= count, we revert to previous state)
                        # Wait, count was decremented by 1.
                        # If we add 0, count is count-1.
                        # If we add 1, count is count.
                        # If we add >1, count increases.
                        # So if new_count <= count, we revert.
                        # Reverting means in_set[u]=True, free_cands=False.
                        # count goes back to original count.
                        pass
                else:
                    # No candidates freed, revert
                    in_set[u] = True
                    count = current_count + 1

        if count > best_count:
            best_count = count
            best_set = in_set.copy()
            
    # Combine results
    V_t2_sel = V_t2[best_set]
    result = np.vstack([V_t1, V_t2_sel])
    
    return result
