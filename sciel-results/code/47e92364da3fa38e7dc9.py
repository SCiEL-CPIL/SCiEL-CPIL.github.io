import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    # Compute conflicts
    # Dot product > 2 is a conflict
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    best_selected = np.zeros(N4, dtype=bool)
    best_count = 0
    
    rng = np.random.default_rng(42)
    
    # Run multiple greedy MIS attempts
    max_restarts = 100 # Reduced to allow time for local search
    
    for r in range(max_restarts):
        if time.time() > deadline - 20: # Reserve 20s for local search of best
            break
            
        alive = np.ones(N4, dtype=bool)
        selected_indices = []
        degrees = conflicts.sum(axis=1).astype(np.int32).copy()
        
        while np.any(alive) and time.time() < deadline - 25:
            degs_masked = np.where(alive, degrees, np.inf)
            min_idx = np.argmin(degs_masked)
            
            if not alive[min_idx]:
                break
                
            selected_indices.append(min_idx)
            
            neighbors = np.where(conflicts[min_idx])[0]
            to_remove = np.unique(np.concatenate(([min_idx], neighbors)))
            
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= (conflicts[rem][:, to_remove] == 1).sum(axis=1)
                
        # Local Search on the selected set
        selected_mask = np.zeros(N4, dtype=bool)
        selected_mask[selected_indices] = True
        unselected_mask = ~selected_mask
        
        # 1. Try to Add any compatible unselected vector
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            # Vectorized check: for each v, is conflicts[v] & selected_mask empty?
            # conflicts @ selected_mask gives count of conflicts with selected
            # We want count == 0
            conflict_counts = (conflicts & selected_mask[None, :]).sum(axis=1)
            # Actually, conflicts is bool. 
            # (conflicts & selected_mask[None, :]) is not right broadcasting.
            # conflicts is (N, N). selected_mask is (N,).
            # We want for each row i: sum(conflicts[i, j] for j where selected_mask[j])
            # This is conflicts.dot(selected_mask) if conflicts is int.
            # Let's use int matrix for speed? Or just boolean sum.
            
            # Efficient check:
            # If selected_mask is bool, conflicts @ selected_mask (with conflicts as int)
            # Let's keep conflicts as bool, but use np.dot with int conversion?
            # Or just:
            # valid_adds = np.all(~(conflicts & selected_mask[None, :]), axis=1)
            # This is O(N^2).
            
            # Faster:
            # conflict_counts = conflicts.dot(selected_mask.astype(np.int32))
            # valid_adds = (conflict_counts == 0) & unselected_mask
            
            conflict_counts = conflicts.dot(selected_mask) # conflicts is bool, treated as 0/1
            valid_adds = (conflict_counts == 0) & unselected_mask
            
            if np.any(valid_adds):
                # Add all valid adds? Or one by one?
                # Adding all at once might be invalid if they conflict with each other.
                # But if they are all compatible with current S, do they conflict with each other?
                # Yes, they might.
                # So we should add one, update, repeat.
                # Or pick one randomly.
                add_indices = np.where(valid_adds)[0]
                # Pick one
                idx = add_indices[0] 
                selected_mask[idx] = True
                unselected_mask[idx] = False
                changed = True
                
        # 2. Try Swap-to-Add
        # We want to find u in S, v in U such that v conflicts only with u in S.
        # And after swapping u->v, we can add w.
        # Simplified: Just try to find a swap that increases "potential" or allows add.
        # For now, let's just try to find a swap that allows an immediate add.
        
        swapped = True
        while swapped and time.time() < deadline - 5:
            swapped = False
            S_indices = np.where(selected_mask)[0]
            if len(S_indices) == 0: break
            
            # Try each u in S
            for u in S_indices:
                if time.time() > deadline - 5: break
                
                # S minus u
                S_minus_u = selected_mask.copy()
                S_minus_u[u] = False
                
                # Find v in U such that v conflicts only with u (i.e., no conflict with S_minus_u)
                # conflict_counts_v = conflicts.dot(S_minus_u)
                # We want conflict_counts_v == 0 for v in U
                conflict_counts_v = conflicts.dot(S_minus_u)
                valid_swaps = (conflict_counts_v == 0) & unselected_mask
                
                if np.any(valid_swaps):
                    v_indices = np.where(valid_swaps)[0]
                    # Try each v? Or just one?
                    # Let's try the first one
                    v = v_indices[0]
                    
                    # Tentatively swap
                    selected_mask[u] = False
                    selected_mask[v] = True
                    unselected_mask[u] = True
                    unselected_mask[v] = False
                    
                    # Try Add
                    conflict_counts_new = conflicts.dot(selected_mask)
                    valid_adds_new = (conflict_counts_new == 0) & unselected_mask
                    
                    if np.any(valid_adds_new):
                        # Commit swap and add
                        add_idx = np.where(valid_adds_new)[0][0]
                        selected_mask[add_idx] = True
                        unselected_mask[add_idx] = False
                        swapped = True # We made progress (size increased by 1 effectively, or at least changed state)
                        # Actually, size increased by 1 (add) - 1 (swap out) + 1 (swap in) = +1?
                        # Start: |S|. Swap: |S|. Add: |S|+1.
                        # So yes, size increased.
                    else:
                        # Revert swap
                        selected_mask[u] = True
                        selected_mask[v] = False
                        unselected_mask[u] = False
                        unselected_mask[v] = True
                        # Try next v or next u
                        continue
                
        count = np.sum(selected_mask)
        if count > best_count:
            best_count = count
            best_selected = selected_mask.copy()

    result_v4 = vectors_v4[best_selected]
    final_vectors = np.vstack([result_v4, vectors_v1])
    return final_vectors
