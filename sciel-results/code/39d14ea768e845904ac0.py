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
    
    best_count = 0
    best_selected = np.zeros(n, dtype=np.uint8)
    
    # 3. Multi-Start Search
    # Strategy:
    # A) Randomized Greedy Construction (Min-Degree heuristic)
    # B) Vectorized Local Search Refinement (1-for-2 swaps primarily)
    
    rng = np.random.default_rng(42)
    
    # Estimate time per restart to determine max restarts
    # We'll run at least 10 restarts to estimate, then adjust.
    
    def greedy_construct():
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
            deg = (conflict * remaining[np.newaxis, :]).sum(axis=1)
            deg[remaining == 0] = n + 1
            
            min_deg = deg.min()
            if min_deg >= n:
                break
                
            # Pick randomly from those with min degree
            # To add diversity, we can pick from the top 5% min-degree nodes
            # or just min-degree. Let's stick to min-degree for stability.
            candidates_to_pick = np.where((deg == min_deg) & (remaining > 0))[0]
            if len(candidates_to_pick) == 0:
                break
                
            pick_idx = rng.choice(candidates_to_pick)
            
            selected[pick_idx] = 1
            remaining[pick_idx] = 0
            # Update remaining: remove neighbors of pick_idx
            remaining[conflict[pick_idx] == 1] = 0
            count += 1
            
        return selected

    def local_search(selected):
        sel_indices = np.where(selected == 1)[0]
        if len(sel_indices) == 0:
            return selected, 0
            
        improved = True
        max_moves = 200
        move_count = 0
        moves_made = 0
        
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
            
            # base_mask: nodes compatible with ALL S
            base_mask = (total_inc == 0) & (selected == 0)
            
            # Try 1-for-2 moves
            # For each v in S, find u, w compatible with S\{v} and with each other
            # We iterate through S in a random order to avoid bias
            order = rng.permutation(k_sel)
            
            for idx in order:
                if time.time() > deadline - 5.0:
                    break
                    
                v_idx_in_S = idx
                v = sel_indices[v_idx_in_S]
                
                # Construct Cand_v: nodes compatible with S \ {v}
                # x is in Cand_v if:
                # (total_inc[x] == 0 and not selected[x]) OR
                # (total_inc[x] == 1 and conflict_with_S[x, v_idx_in_S] == 1 and not selected[x])
                
                cond_mask = (total_inc == 1) & (conflict_with_S[:, v_idx_in_S] == 1) & (selected == 0)
                Cand_v = base_mask | cond_mask
                
                cand_indices = np.where(Cand_v > 0)[0]
                
                if len(cand_indices) < 2:
                    continue
                
                # Check if there is any compatible pair in cand_indices
                # sub_conflict[u, w] == 0 means compatible
                sub_conflict = conflict[np.ix_(cand_indices, cand_indices)]
                # We need off-diagonal 0s
                # Note: diagonal is 0 in conflict matrix, but we don't care about self
                # We check if any entry is 0.
                # Since conflict is uint8, 0 is compatible.
                
                # Optimization: Only check upper triangle? Or just any.
                # If any 0 exists in sub_conflict, we have a compatible pair.
                # But wait, sub_conflict includes the diagonal which is 0.
                # We need to exclude the diagonal or check for pairs i != j.
                
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
                        moves_made += 1
                        break # Restart while loop with new S
                        
            # If no 1-for-2 found, try 1-for-1 swaps to escape local optima
            # This doesn't increase size but might enable future 1-for-2
            if not improved:
                # Find a v in S and a u not in S such that u is compatible with S\{v}
                # and u has a "better" property (e.g., higher degree in remaining)
                # For simplicity, we can just do a random swap if it maintains validity
                # but we must ensure we don't loop.
                # Let's try to replace v with u if u is compatible with S\{v}
                # This is a neutral move in terms of size, but changes the landscape.
                
                # To avoid infinite loops of neutral swaps, we limit them or use a temperature-like decay.
                # Given time constraints, we might skip this if 1-for-2 is not frequent.
                # However, the inspiration suggests 1-for-1/2/3.
                # Let's implement a simple 1-for-1:
                # Pick a random v in S.
                # Find all u compatible with S\{v}.
                # If any exist, pick one randomly and swap.
                
                if k_sel > 0:
                    v_idx_in_S = rng.integers(k_sel)
                    v = sel_indices[v_idx_in_S]
                    
                    cond_mask = (total_inc == 1) & (conflict_with_S[:, v_idx_in_S] == 1) & (selected == 0)
                    base_mask_v = (total_inc == 0) & (selected == 0)
                    Cand_v = base_mask_v | cond_mask
                    
                    cand_indices = np.where(Cand_v > 0)[0]
                    
                    if len(cand_indices) > 0:
                        u = cand_indices[rng.integers(len(cand_indices))]
                        # Swap v with u
                        selected[v] = 0
                        selected[u] = 1
                        improved = True
                        moves_made += 1
                        # We don't break here immediately to allow more moves in the same pass?
                        # No, restarting the while loop is safer to recompute total_inc.
                        break
        
        return selected, moves_made

    num_restarts = 500 # Upper bound, will be limited by time
    times = []
    
    for restart in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
            
        # Phase 1: Greedy
        selected = greedy_construct()
        
        # Phase 2: Local Search
        selected, moves = local_search(selected)
        
        current_count = selected.sum()
        if current_count > best_count:
            best_count = current_count
            best_selected = selected.copy()
            
        # Track time for adaptive restarts?
        # For now, just let the time guard break the loop.

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
