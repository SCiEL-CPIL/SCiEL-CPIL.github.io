"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # 1. Generate candidate pool
    # We focus on vectors with small integer entries to maximize density.
    # Norms 1, 2, 3, 4 are the most promising.
    candidates = []
    
    # Helper to add vector
    def add_vec(v):
        # Ensure integer type
        v = np.array(v, dtype=np.int64)
        candidates.append(v)

    # Norm 1: +/- e_i
    for i in range(D):
        for s in [1, -1]:
            v = np.zeros(D); v[i] = s
            add_vec(v)
            
    # Norm 2: +/- e_i +/- e_j (i != j)
    for i in range(D):
        for j in range(i+1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(D); v[i] = si; v[j] = sj
                    add_vec(v)
                    
    # Norm 3: +/- e_i +/- e_j +/- e_k (i != j != k)
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for si in [1, -1]:
                    for sj in [1, -1]:
                        for sk in [1, -1]:
                            v = np.zeros(D); v[i] = si; v[j] = sj; v[k] = sk
                            add_vec(v)
                            
    # Norm 4: 
    # Type A: +/- 2 e_i
    for i in range(D):
        for s in [1, -1]:
            v = np.zeros(D); v[i] = 2*s
            add_vec(v)
    # Type B: +/- 2 e_i +/- 2 e_j
    for i in range(D):
        for j in range(i+1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(D); v[i] = 2*si; v[j] = 2*sj
                    add_vec(v)
    # Type C: +/- e_i +/- e_j +/- e_k +/- e_l (4 distinct indices)
    # This generates a lot of vectors. C(11,4)*16 = 5280.
    for indices in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D)
            for idx, s in zip(indices, signs):
                v[idx] = s
            add_vec(v)
            
    candidates = np.array(candidates)
    n_cand = len(candidates)
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # Precompute pairwise dot products for all candidates?
    # n_cand is approx:
    # Norm 1: 22
    # Norm 2: 220
    # Norm 3: 1980
    # Norm 4: 22 + 220 + 5280 = 5522
    # Total: ~7744
    # Matrix 7744x7744 is ~60MB (float64). This is feasible.
    # Precomputing the full dot product matrix allows O(1) compatibility checks.
    
    # However, we need to check compatibility of a candidate with the *current set*.
    # If we keep a boolean mask of selected candidates, we can just check the subset.
    # Let's precompute the full dot matrix.
    
    dot_matrix = candidates @ candidates.T
    
    # 2. Simulated Annealing on the Independent Set Problem
    # State: boolean array `selected` of size n_cand.
    # Objective: Maximize sum(selected).
    # Constraint: For all i, j where selected[i] and selected[j] are True,
    #             2 * dot_matrix[i, j] <= min(norms_sq[i], norms_sq[j])
    
    # We will use a swap-based SA.
    # Move: Pick one selected vector to remove, and one unselected vector to add.
    # Check if the new set is valid.
    
    # To speed up validity checks, we can precompute for each candidate i,
    # the list of candidates j that are INCOMPATIBLE with i.
    # Incompatible if 2 * dot[i,j] > min(norms_sq[i], norms_sq[j])
    
    # Let's create a conflict matrix (boolean).
    # conflict[i, j] is True if i and j cannot both be in the set.
    
    # Note: A vector is always compatible with itself? No, we only pick distinct vectors.
    # But the condition 2*v.v <= v.v => v.v <= 0 is false for nonzero v.
    # So diagonal is True in conflict matrix if we consider self-conflict, but we don't.
    
    min_norms = np.minimum(norms_sq[:, None], norms_sq[None, :])
    conflict = 2 * dot_matrix > min_norms
    
    # Set diagonal to False (a vector doesn't conflict with itself in the context of picking it once)
    # Actually, we just won't pick the same index twice.
    np.fill_diagonal(conflict, False)
    
    # 3. Initialize SA
    # Start with a greedy solution.
    # Order candidates by some heuristic? Maybe by norm?
    # Let's try random initialization with greedy fill.
    
    best_selected = np.zeros(n_cand, dtype=bool)
    best_count = 0
    
    rng = np.random.default_rng(42)
    
    # Greedy init
    # Try a few random orders
    for _ in range(10):
        order = rng.permutation(n_cand)
        curr_selected = np.zeros(n_cand, dtype=bool)
        curr_count = 0
        for idx in order:
            # Check if idx is compatible with all currently selected
            # We need to check if conflict[idx, curr_selected_indices] has any True.
            # Since n_cand is ~7700, curr_selected is a boolean array.
            # conflict[idx] is a boolean array.
            # If np.any(conflict[idx] & curr_selected): then conflict exists.
            if not np.any(conflict[idx] & curr_selected):
                curr_selected[idx] = True
                curr_count += 1
        
        if curr_count > best_count:
            best_count = curr_count
            best_selected = curr_selected.copy()
            
    # SA Loop
    # We want to improve the count.
    # Temperature T controls probability of accepting worse solutions (lower count or invalid? No, we only consider valid moves).
    # Actually, standard SA on Independent Set usually allows invalid states with a penalty, or only valid moves.
    # If we only allow valid moves (swap that results in a valid set), we are doing a random walk on the space of valid independent sets.
    # We want to move to larger sets.
    # Move: Remove i (selected), Add j (unselected).
    # Valid if:
    #   1. j is not in conflict with any selected k (k != i).
    #   2. (Implicitly, i was compatible with the rest, which it is).
    #   Wait, if we remove i, the set is valid. Then we add j.
    #   We need j to be compatible with all (selected \ {i}).
    #   This is equivalent to: j is not in conflict with any selected k, EXCEPT possibly i.
    #   But since i is being removed, we don't care about conflict(j, i).
    #   So check: np.any(conflict[j] & curr_selected) == 0 OR (the only conflict is with i).
    #   Actually, simpler: 
    #   Let S be current set.
    #   We want to add j and remove i.
    #   New set S' = (S \ {i}) U {j}.
    #   Valid if for all k in S' \ {j}, conflict[j, k] is False.
    #   S' \ {j} = S \ {i}.
    #   So we need: for all k in S, k != i, conflict[j, k] is False.
    #   This means: conflict[j] & (curr_selected - e_i) must be all False.
    #   Where e_i is a one-hot vector.
    #   Equivalent to: (conflict[j] & curr_selected).sum() <= (1 if conflict[j, i] else 0).
    #   Actually, if conflict[j, i] is True, then the sum of conflicts in S is at most 1 (the one with i).
    #   If conflict[j, i] is False, then the sum of conflicts in S must be 0.
    
    T = 1.0
    T_min = 1e-6
    alpha = 0.9995
    
    curr_selected = best_selected.copy()
    curr_count = best_count
    
    # Precompute indices of selected and unselected for faster sampling?
    # Or just sample randomly.
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Pick a random selected index i
        selected_indices = np.where(curr_selected)[0]
        if len(selected_indices) == 0:
            break
        i = selected_indices[rng.integers(0, len(selected_indices))]
        
        # Pick a random unselected index j
        unselected_indices = np.where(~curr_selected)[0]
        if len(unselected_indices) == 0:
            break
        j = unselected_indices[rng.integers(0, len(unselected_indices))]
        
        # Check validity of swap (i -> j)
        # Condition: j is compatible with all k in S \ {i}
        # Let C_j = conflict[j] (boolean array)
        # Let S_curr = curr_selected (boolean array)
        # We need C_j & (S_curr - e_i) to be all False.
        # Let's compute the number of conflicts of j with S_curr.
        n_conflicts = np.sum(C_j & S_curr)
        
        # If conflict[j, i] is True, then one of these conflicts is with i.
        # So the number of conflicts with S \ {i} is n_conflicts - (1 if conflict[j, i] else 0).
        # This number must be 0.
        
        if conflict[j, i]:
            valid_swap = (n_conflicts == 1)
        else:
            valid_swap = (n_conflicts == 0)
            
        if valid_swap:
            # Perform swap
            curr_selected[i] = False
            curr_selected[j] = True
            # Count doesn't change!
            # Wait, this is a swap. The size of the independent set remains the same.
            # This is a random walk on fixed-size independent sets.
            # How do we grow the set?
            
            # We need a move that INCREASES the size.
            # Move: Add j if j is compatible with ALL selected.
            # If we can find such a j, we accept it (always improves).
            
            # Let's modify the move generation:
            # 1. Try to add a random unselected j.
            #    If valid (compatible with all selected), accept (count + 1).
            # 2. If not, try to swap a random selected i with a random unselected j.
            #    If valid, accept (count stays same, but we move in the space).
            # 3. If not, try to remove a random selected i? (Count - 1, only if T allows).
            
            # Let's restructure the loop.
            
            # Rollback the swap for now, we'll do it properly in the next iteration of the logic.
            curr_selected[i] = True
            curr_selected[j] = False
            
        # --- Revised Move Logic ---
        
        # Strategy:
        # 1. Attempt to insert a random unselected vector.
        # 2. If that fails, attempt a swap.
        # 3. If that fails, attempt a removal (with probability based on T).
        
        # 1. Insert
        if len(unselected_indices) > 0:
            j = unselected_indices[rng.integers(0, len(unselected_indices))]
            n_conflicts = np.sum(conflict[j] & curr_selected)
            if n_conflicts == 0:
                curr_selected[j] = True
                curr_count += 1
                if curr_count > best_count:
                    best_count = curr_count
                    best_selected = curr_selected.copy()
                # Reset T? Or just continue.
                continue
                
        # 2. Swap
        # Pick i from selected, j from unselected.
        # We already picked i and j above? No, let's pick fresh.
        i = selected_indices[rng.integers(0, len(selected_indices))]
        j = unselected_indices[rng.integers(0, len(unselected_indices))]
        
        # Check swap validity
        # j must be compatible with S \ {i}
        # n_conflicts_j_in_S = np.sum(conflict[j] & curr_selected)
        # If conflict[j, i] is True, we need n_conflicts_j_in_S == 1.
        # Else, we need n_conflicts_j_in_S == 0.
        
        n_conf_j = np.sum(conflict[j] & curr_selected)
        if conflict[j, i]:
            swap_valid = (n_conf_j == 1)
        else:
            swap_valid = (n_conf_j == 0)
            
        if swap_valid:
            curr_selected[i] = False
            curr_selected[j] = True
            # Count unchanged
            # We accept this move to explore the space.
            # We don't update best_count here as it didn't change.
            continue
            
        # 3. Removal (if stuck)
        # Remove a random selected vector.
        # This decreases count. Accept with prob exp(-delta/T) where delta = -1 (improvement in objective if we minimize -count).
        # Here we maximize count. Removing decreases count by 1.
        # Prob = exp(-1/T).
        if T > 0.1: # Only try removals when T is high enough
            i = selected_indices[rng.integers(0, len(selected_indices))]
            # Check if removing i is valid? Yes, any subset of a valid set is valid.
            # But does it help? It might allow future insertions.
            # Acceptance probability for a move that decreases objective (count) by 1.
            # delta_obj = -1.
            # p = exp(-(-1)/T) = exp(1/T). This is always > 1.
            # Wait, standard SA: p = exp(-Delta / T).
            # If we maximize, Delta = new_obj - old_obj.
            # Removal: new_obj = count - 1. Delta = -1.
            # p = exp(1/T). Always accept?
            # Usually we want to accept worse moves with decreasing probability.
            # If we remove, we get worse. So we should accept with prob exp(-1/T).
            
            r = rng.random()
            if r < np.exp(-1.0 / T):
                curr_selected[i] = False
                curr_count -= 1
                
        T *= alpha

    return candidates[best_selected]

# EVOLVE-BLOCK-END
