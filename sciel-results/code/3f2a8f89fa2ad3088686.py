import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    D = 11
    
    # 1. Generate Candidate Pool: Weight-4 vectors
    # Norm^2 = 4. Max norm in set is 2.
    # Constraint: ||v_i - v_j||^2 >= 4 for all i != j.
    # For weight-4 vectors, this means dot(v_i, v_j) <= 2.
    # We generate all weight-4 vectors in R^11.
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
    
    best_count = 0
    best_selected = np.zeros(n, dtype=np.uint8)
    
    # 3. Multi-Start Search
    # Strategy:
    # A) Randomized Greedy Construction (Min-Degree heuristic)
    # B) Vectorized Local Search Refinement (1-for-2 swaps and greedy adds)
    
    rng = np.random.default_rng(42)
    
    def greedy_construct(seed):
        """Randomized min-degree greedy construction."""
        local_rng = np.random.default_rng(seed)
        selected = np.zeros(n, dtype=np.uint8)
        remaining = np.ones(n, dtype=np.uint8)
        
        while time.time() < deadline - 10.0:
            rem_idx = np.where(remaining > 0)[0]
            if len(rem_idx) == 0:
                break
                
            # Calculate effective degree in the remaining subgraph
            deg = (conflict * remaining[np.newaxis, :]).sum(axis=1)
            deg[remaining == 0] = n + 1
            
            min_deg = deg.min()
            if min_deg >= n:
                break
                
            # Pick randomly from those with min degree
            candidates_to_pick = np.where((deg == min_deg) & (remaining > 0))[0]
            if len(candidates_to_pick) == 0:
                break
                
            pick_idx = local_rng.choice(candidates_to_pick)
            
            selected[pick_idx] = 1
            remaining[pick_idx] = 0
            # Update remaining: remove neighbors of pick_idx
            remaining[conflict[pick_idx] == 1] = 0
            
        return selected

    def local_search(selected, seed):
        """
        Vectorized local search.
        1. Try to add any compatible vector.
        2. Try 1-for-2 swaps (remove v, add u, w).
        """
        local_rng = np.random.default_rng(seed)
        improved = True
        max_moves = 1000
        move_count = 0
        
        while improved and move_count < max_moves and time.time() < deadline - 5.0:
            improved = False
            move_count += 1
            
            sel_indices = np.where(selected == 1)[0]
            k_sel = len(sel_indices)
            
            if k_sel == 0:
                break
            
            # Compute Total_Incompatibility for all nodes against current S
            # conflict[:, sel_indices] gives incompatibility matrix for S
            conflict_with_S = conflict[:, sel_indices]
            total_inc = conflict_with_S.sum(axis=1)
            
            # 1. Try to add a vertex (if any)
            # A vertex x is addable if total_inc[x] == 0 and selected[x] == 0
            addable_mask = (total_inc == 0) & (selected == 0)
            addable_indices = np.where(addable_mask)[0]
            
            if len(addable_indices) > 0:
                # Pick one (random or min-degree in remaining, but random is fine for diversity)
                # To be slightly smarter, pick one that minimizes future conflicts?
                # For now, random pick from addable to avoid bias.
                best_add = local_rng.choice(addable_indices)
                selected[best_add] = 1
                improved = True
                continue # Restart loop with new S
            
            # 2. Try 1-for-2 moves
            # For each v in S, find u, w compatible with S\{v} and with each other
            # We iterate through S in a random order to avoid bias
            order = local_rng.permutation(k_sel)
            
            for idx in order:
                if time.time() > deadline - 5.0:
                    break
                    
                v_idx_in_S = idx
                v = sel_indices[v_idx_in_S]
                
                # Construct Cand_v: nodes compatible with S \ {v}
                # x is in Cand_v if:
                # (total_inc[x] == 0 and not selected[x]) OR
                # (total_inc[x] == 1 and conflict_with_S[x, v_idx_in_S] == 1 and not selected[x])
                
                # Note: total_inc[x] counts conflicts with ALL S.
                # If total_inc[x] == 1 and that conflict is with v, then x is compatible with S\{v}.
                # If total_inc[x] == 0, x is compatible with S, so also S\{v}.
                
                cond_mask = (total_inc == 1) & (conflict_with_S[:, v_idx_in_S] == 1) & (selected == 0)
                base_mask = (total_inc == 0) & (selected == 0)
                Cand_v = base_mask | cond_mask
                
                cand_indices = np.where(Cand_v > 0)[0]
                
                if len(cand_indices) < 2:
                    continue
                
                # Check if there is any compatible pair in cand_indices
                # sub_conflict[u, w] == 0 means compatible
                sub_conflict = conflict[np.ix_(cand_indices, cand_indices)]
                
                # We need off-diagonal 0s.
                # Mask diagonal to 1 to ignore self-compatibility
                sub_conflict_check = sub_conflict.copy()
                np.fill_diagonal(sub_conflict_check, 1)
                
                if (sub_conflict_check == 0).any():
                    zero_rows, zero_cols = np.where(sub_conflict_check == 0)
                    if len(zero_rows) > 0:
                        u_local = zero_rows[0]
                        w_local = zero_cols[0]
                        u = cand_indices[u_local]
                        w = cand_indices[w_local]
                        
                        # Perform move: remove v, add u, w
                        selected[v] = 0
                        selected[u] = 1
                        selected[w] = 1
                        improved = True
                        break # Restart while loop with new S
            
            # 3. If no 1-for-2 found, try 1-for-1 swaps to escape local optima?
            # The inspiration suggests this can help, but 1-for-2 is the primary driver.
            # We can add a simple random 1-for-1 if we are stuck, but it's risky.
            # Let's stick to the proven 1-for-2 and greedy add.
            
        return selected

    # Run multiple restarts
    num_restarts = 20 # Adjust based on time
    
    for r in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Greedy
        sel = greedy_construct(seed=r*1000 + 42)
        
        # Local Search
        sel = local_search(sel, seed=r*2000 + 42)
        
        count = np.sum(sel)
        if count > best_count:
            best_count = count
            best_selected = sel.copy()
            
        # Early stopping if we hit a very high count? 
        # The theoretical max for weight-4 in D=11 is unknown but likely < 2000.
        # If we get > 1500, it's very good.

    if best_count == 0:
        # Fallback: return a single valid vector
        return np.array([[1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)

    result = candidates[best_selected]
    return result.astype(np.int64)
