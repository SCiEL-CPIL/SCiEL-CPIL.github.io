import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    D = 11
    
    # 1. Generate Candidate Pool
    # We focus on weight-4 vectors (norm^2 = 4).
    # Constraint: ||v_i - v_j||^2 >= 4 (since max norm is 2, max norm^2 is 4).
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j = 4 + 4 - 2*v_i.v_j = 8 - 2*v_i.v_j.
    # We need 8 - 2*v_i.v_j >= 4  =>  2*v_i.v_j <= 4  =>  v_i.v_j <= 2.
    # So, two weight-4 vectors are compatible if their dot product is <= 2.
    
    candidates_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)
    
    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)
    
    # 2. Precompute Conflict Matrix
    # conflict[i, j] = 1 if dot(candidates[i], candidates[j]) > 2
    c_int = candidates.astype(np.int32)
    dot = c_int @ c_int.T
    
    # Conflict if dot > 2
    conflict = (dot > 2).astype(np.uint8)
    np.fill_diagonal(conflict, 0)
    
    # Compatibility matrix (1 if compatible)
    compat = 1 - conflict
    np.fill_diagonal(compat, 1)
    
    # Precompute degrees
    degrees = conflict.sum(axis=1)
    
    best_count = 0
    best_selected = np.zeros(n, dtype=np.uint8)
    
    # 3. Multi-Start Search
    # Strategy:
    # A) Randomized Greedy Construction (Min-Degree heuristic)
    # B) Local Search Refinement (1-for-2, 1-for-3 moves)
    
    num_restarts = 200
    rng = np.random.default_rng(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
            
        # --- Phase 1: Randomized Greedy Construction ---
        selected = np.zeros(n, dtype=np.uint8)
        remaining = np.ones(n, dtype=np.uint8)
        
        count = 0
        while True:
            if time.time() > deadline - 10.0:
                break
                
            rem_idx = np.where(remaining > 0)[0]
            if len(rem_idx) == 0:
                break
                
            # Calculate effective degree in the remaining subgraph
            # deg[i] = sum(conflict[i, j] for j in remaining)
            deg = (conflict * remaining[np.newaxis, :]).sum(axis=1)
            deg[remaining == 0] = n + 1 # Mask out non-remaining
            
            min_deg = deg.min()
            if min_deg >= n:
                break
                
            # Pick randomly from those with min degree
            candidates_to_pick = np.where((deg == min_deg) & (remaining > 0))[0]
            if len(candidates_to_pick) == 0:
                break
                
            pick_idx = rng.choice(candidates_to_pick)
            
            selected[pick_idx] = 1
            remaining[pick_idx] = 0
            # Update remaining: remove neighbors of pick_idx
            remaining[conflict[pick_idx] == 1] = 0
            count += 1
            
        # --- Phase 2: Local Search Refinement ---
        sel_indices = np.where(selected == 1)[0]
        if len(sel_indices) == 0:
            continue
            
        improved = True
        max_moves = 100
        move_count = 0
        
        while improved and move_count < max_moves and time.time() < deadline - 5.0:
            improved = False
            move_count += 1
            
            # Shuffle selected indices to avoid bias
            order = rng.permutation(len(sel_indices))
            
            for i in order:
                if time.time() > deadline - 5.0:
                    break
                    
                v = sel_indices[i]
                
                # Compute mask of vertices compatible with (S \ {v})
                mask_except_v = np.ones(n, dtype=np.uint8)
                for s in sel_indices:
                    if s != v:
                        mask_except_v = mask_except_v & compat[s]
                
                mask_except_v[sel_indices] = 0 # Exclude all selected
                
                cand_indices = np.where(mask_except_v > 0)[0]
                
                if len(cand_indices) < 2:
                    continue
                    
                # Try to find a compatible pair (u, w) in cand_indices
                # This is finding an edge in the compat graph induced by cand_indices.
                # Or, finding a non-edge in the conflict graph.
                
                # To be faster, we can just check if the induced subgraph has any 0 in conflict.
                # We don't need to find the specific pair immediately if we just want to know if a move is possible.
                # But we need to pick the pair.
                
                # Let's try to pick the pair that is "best" (e.g., min degree in remaining graph).
                # For now, just find any pair.
                
                # Get conflict submatrix
                c_sub = conflict[np.ix_(cand_indices, cand_indices)]
                
                # Find first compatible pair
                # Use np.argwhere? Slow.
                # Use:
                # mask = c_sub == 0
                # np.fill_diagonal(mask, False)
                # if mask.any():
                #    u_idx, w_idx = np.argwhere(mask)[0]
                
                # Faster:
                # For each u, check if it has any compatible w.
                # This is O(|C|^2).
                
                # Let's try a heuristic: pick a random u, check if it has compatible w.
                # Repeat a few times.
                
                found_pair = False
                # Try a few random candidates
                for _ in range(10):
                    if len(cand_indices) == 0:
                        break
                    u = cand_indices[rng.integers(len(cand_indices))]
                    # Find w compatible with u
                    compat_w = compat[u, cand_indices]
                    compat_w[cand_indices == u] = 0 # Exclude self
                    
                    if compat_w.any():
                        w = cand_indices[np.argmax(compat_w)]
                        # Perform move: Remove v, add u, w
                        selected[v] = 0
                        selected[u] = 1
                        selected[w] = 1
                        
                        # Update sel_indices
                        sel_indices = np.where(selected == 1)[0]
                        
                        improved = True
                        found_pair = True
                        break
                        
                if found_pair:
                    break # Restart loop with new state
                
                # Try 1-for-3 move if 1-for-2 failed
                if not improved and len(cand_indices) >= 3:
                    # Find a compatible triple
                    # This is harder.
                    # Pick u, then find w compatible with u, then find x compatible with u and w.
                    
                    found_triple = False
                    for _ in range(5):
                        if len(cand_indices) == 0:
                            break
                        u = cand_indices[rng.integers(len(cand_indices))]
                        compat_u = compat[u, cand_indices]
                        compat_u[cand_indices == u] = 0
                        
                        if compat_u.sum() < 2:
                            continue
                        
                        # Pick w
                        w = cand_indices[np.argmax(compat_u)]
                        
                        # Find x compatible with both u and w
                        compat_ux = compat_u & compat[w, cand_indices]
                        compat_ux[cand_indices == u] = 0
                        compat_ux[cand_indices == w] = 0
                        
                        if compat_ux.any():
                            x = cand_indices[np.argmax(compat_ux)]
                            
                            # Perform move
                            selected[v] = 0
                            selected[u] = 1
                            selected[w] = 1
                            selected[x] = 1
                            
                            sel_indices = np.where(selected == 1)[0]
                            improved = True
                            found_triple = True
                            break
                            
                    if found_triple:
                        break
        
        # Final count
        current_count = selected.sum()
        if current_count > best_count:
            best_count = current_count
            best_selected = selected.copy()
            
    # 4. Extract Result
    sel_indices = np.where(best_selected == 1)[0]
    if len(sel_indices) == 0:
        # Fallback
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)
        
    result = candidates[sel_indices.astype(np.int64)]
    return result.astype(np.int64)

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} vectors")
    # Verify
    if len(res) > 1:
        norms = np.sum(res**2, axis=1)
        max_norm = np.max(norms)
        dots = res @ res.T
        np.fill_diagonal(dots, 0)
        # dist^2 = ni + nj - 2*dot
        # We need dist^2 >= max_norm
        # ni + nj - 2*dot >= max_norm
        # Since ni=nj=4 for all, 8 - 2*dot >= 4 => 2*dot <= 4 => dot <= 2
        min_dist_sq = np.inf
        for i in range(len(res)):
            for j in range(i+1, len(res)):
                d2 = norms[i] + norms[j] - 2*dots[i,j]
                if d2 < min_dist_sq:
                    min_dist_sq = d2
        print(f"Max norm: {max_norm}, Min dist^2: {min_dist_sq}")
        if min_dist_sq >= max_norm:
            print("Valid")
        else:
            print("INVALID")
