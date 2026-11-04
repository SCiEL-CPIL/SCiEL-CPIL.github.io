"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product

def _generate_candidate_pool():
    """
    Generate a diverse pool of candidate integer vectors in R^11.
    We include shells up to weight 4 to balance density and packing constraints.
    Shell 1: 11 vectors (norm 1)
    Shell 2: 220 vectors (norm sqrt(2))
    Shell 3: 1716 vectors (norm sqrt(3))
    Shell 4: 5280 vectors (norm 2)
    Total ~ 7227 vectors.
    """
    vecs = []
    
    # Shell 1: e_i
    for i in range(D):
        v = np.zeros(D, dtype=np.int64)
        v[i] = 1
        vecs.append(v)
        
    # Shell 2: e_i - e_j
    for i in range(D):
        for j in range(D):
            if i == j: continue
            v = np.zeros(D, dtype=np.int64)
            v[i] = 1
            v[j] = -1
            vecs.append(v)
            
    # Shell 3: Weight 3 vectors with entries in {1, -1}.
    for comb in combinations(range(D), 3):
        for signs in product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            vecs.append(v)
            
    # Shell 4: Weight 4. C(11,4) * 2^4 = 330 * 16 = 5280.
    for comb in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            vecs.append(v)
            
    return np.array(vecs, dtype=np.int64)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s buffer for safety
    
    # 1. Generate candidate pool
    candidates = _generate_candidate_pool()
    n_cand = len(candidates)
    
    # Precompute norms squared
    norms_sq = np.sum(candidates * candidates, axis=1).astype(np.int64)
    
    # 2. Build Conflict Matrix
    # Two vectors u, v conflict if ||u - v||^2 < max(||u||^2, ||v||^2)
    # We want an Independent Set in this graph.
    
    # Compute Gram matrix G = candidates @ candidates.T
    # Use float32 for memory efficiency, but norms are small integers so precision is safe.
    G = (candidates.astype(np.float32) @ candidates.T.astype(np.float32))
    
    ns = norms_sq.astype(np.float32)
    ns_col = ns[:, np.newaxis]
    ns_row = ns[np.newaxis, :]
    max_ns = np.maximum(ns_col, ns_row)
    
    dist_sq = ns_col + ns_row - 2 * G
    conflict = dist_sq < max_ns
    np.fill_diagonal(conflict, False)
    
    # 3. Simulated Annealing
    # Objective: Maximize size of Independent Set.
    # State: Bitmask of selected vectors.
    # Moves: Swap one selected vector with one unselected vector.
    # Energy: -size (we minimize energy, so maximize size).
    # Actually, easier to just track size and accept/reject based on size change and temperature.
    
    # Initial State: Greedy Independent Set
    # Sort by degree (fewest conflicts first)
    degrees = np.sum(conflict, axis=1)
    order = np.argsort(degrees)
    
    selected = np.zeros(n_cand, dtype=bool)
    selected_indices = []
    
    # Greedy initialization
    for v in order:
        if selected[v]:
            continue
        # Check if v conflicts with any already selected
        # neighbors of v are conflict[v, :]
        # We need to check if any neighbor is in selected
        # This is O(N) per check, total O(N^2) which is 50M ops. Acceptable.
        # Optimization: Maintain a 'blocked' array? 
        # Simpler: Just check intersection.
        if not np.any(selected & conflict[v, :]):
            selected[v] = True
            selected_indices.append(v)
            
    current_size = len(selected_indices)
    best_size = current_size
    best_selected = selected.copy()
    
    # Annealing Parameters
    # T0 should be high enough to allow bad moves initially.
    # Delta E is -1, 0, or +1 (size change).
    # If we swap in a good one and out a bad one, size +1 (always accept).
    # If we swap out a good one and in a bad one, size -1 (accept with prob exp(-1/T)).
    # If neutral, size 0 (accept with prob 1 or exp(0/T)=1).
    
    T0 = 5.0
    T_end = 0.1
    cooling_rate = 0.999
    
    rng = np.random.default_rng(42)
    
    # Precompute lists for faster sampling?
    # We need to pick a random selected index and a random unselected index.
    # Using np.where(selected) is O(N). With N~7000, this is fast enough for ~10k-50k steps.
    
    t = T0
    steps_per_temp = 50 # Adjust based on time
    
    while time.time() < deadline - 5:
        for _ in range(steps_per_temp):
            if time.time() > deadline - 5:
                break
                
            # Get current selected and unselected indices
            sel_idx = np.where(selected)[0]
            unsel_idx = np.where(~selected)[0]
            
            if len(sel_idx) == 0 or len(unsel_idx) == 0:
                break
                
            # Pick random swap
            i = sel_idx[rng.integers(len(sel_idx))]
            j = unsel_idx[rng.integers(len(unsel_idx))]
            
            # Calculate energy change (size change)
            # If we remove i and add j:
            # New size = current_size - 1 + (1 if j is compatible with (selected \ {i}) else 0)
            
            # Check compatibility of j with current selected set excluding i
            # conflict[j, :] gives all conflicts with j.
            # We need to check if any of them are in selected, EXCLUDING i.
            
            # Create a temporary view of selected without i?
            # Or simply: 
            # conflicts_with_j = conflict[j, :]
            # Is there any k in selected such that k != i and conflicts_with_j[k] is True?
            
            # selected_without_i = selected.copy(); selected_without_i[i] = False
            # This is O(N). 
            # Better: 
            # num_conflicts = np.sum(conflict[j, :] & selected)
            # If i is a conflict with j, then num_conflicts includes i.
            # We want to know if there are conflicts OTHER than i.
            
            is_conflict_with_i = conflict[j, i]
            num_conflicts_total = np.sum(conflict[j, :] & selected)
            
            if is_conflict_with_i:
                # i was one of the conflicts. Removing i reduces conflict count by 1.
                num_conflicts_after_removal = num_conflicts_total - 1
            else:
                # i was not a conflict. Removing i doesn't change conflict count for j.
                num_conflicts_after_removal = num_conflicts_total
            
            if num_conflicts_after_removal == 0:
                # j is compatible with the rest. Swap is valid and increases size by 1 (if i was essential? No, i is removed, j is added).
                # Size change: -1 (remove i) + 1 (add j) = 0? 
                # Wait. If j was incompatible with i, but compatible with others, then adding j while keeping i was impossible.
                # By removing i, we free up space.
                # So the new set is valid.
                # Size change is 0? 
                # Let's trace:
                # Old set S. i in S. j not in S.
                # New set S' = S \ {i} U {j}.
                # |S'| = |S|.
                # So size change is 0.
                # However, the *quality* of the set might be better/worse for future moves?
                # In standard MIS, we just care about size.
                # But wait, if j was conflicting with i, then j couldn't be added to S.
                # By swapping, we maintain size.
                # Is there a case where size increases?
                # Only if we could add j WITHOUT removing i. But j is not in S, and S is independent.
                # So j must conflict with at least one element in S.
                # If j conflicts with ONLY i, then removing i allows adding j. Size stays same.
                # If j conflicts with i AND k, then removing i still leaves conflict with k. Size stays same (but set is invalid? No, we check validity).
                # Actually, my check `num_conflicts_after_removal == 0` ensures validity.
                # So any valid swap maintains size.
                # How do we increase size?
                # We can't increase size with a single swap if the set is maximal?
                # If the set is maximal, no unselected node is compatible with the whole set.
                # So any valid swap must remove a node that was blocking the new node.
                # Thus, size is constant during valid swaps.
                
                # This implies Simulated Annealing with swaps only explores the space of MAXIMAL independent sets of the SAME size.
                # It cannot find a LARGER set unless it allows "intermediate" states that are not independent sets?
                # Or we need a different move: Add a node (if possible) or Remove a node.
                # But if we add a node, size increases. If we remove, size decreases.
                # Standard approach for MIS with SA:
                # State: Set of nodes.
                # Moves:
                # 1. Add a node v if it doesn't conflict with any in S. (Size +1, always accept)
                # 2. Remove a node v from S. (Size -1, accept with prob exp(-1/T))
                # 3. Swap v in S with u not in S. (Size 0, accept with prob exp(0/T)=1? Or based on some potential?)
                
                # Let's implement Add/Remove/Swap.
                
                # For now, if valid swap, accept it to explore the landscape.
                selected[i] = False
                selected[j] = True
                # Size unchanged
                # We should track best_size. Since size is same, no update to best_size.
                # But the configuration changes, allowing future moves.
                
            else:
                # Swap is invalid (j conflicts with other nodes in S).
                # We could still move to this invalid state in SA? 
                # Usually SA for combinatorial problems keeps the state valid.
                # If we keep state valid, we can only do valid swaps.
                # This might get stuck in local maxima of size.
                # To escape, we need to allow size decrease (Remove).
                
                # Let's try a different move set:
                # 1. Try to add a random unselected node. If valid, do it (Size +1).
                # 2. Try to remove a random selected node. (Size -1). Accept with prob exp(-1/T).
                # 3. Try a valid swap. (Size 0). Always accept (to shuffle).
                
                # My current loop only does swaps. This is insufficient for size growth.
                # Let's restructure the inner loop.
                
                pass # Placeholder, logic below is better

        # Cooling
        t *= cooling_rate
        if t < T_end:
            break

    # The previous loop structure was flawed for growth. Let's implement a proper SA loop.
    
    # Reset to best found so far (which is the greedy init)
    # Actually, let's restart the SA logic properly.
    
    selected = best_selected.copy()
    current_size = best_size
    
    t = T0
    # Increase steps if time permits?
    # N ~ 7000.
    # Operations per step:
    # 1. np.where (O(N))
    # 2. Random pick
    # 3. Conflict check (O(N) dot product or sum)
    # ~ 3 * 7000 ops = 21k ops.
    # 100k steps = 2.1e9 ops. ~ 2-5 seconds.
    # We have 140s. We can do many more.
    
    # Let's use a more efficient conflict check.
    # Precompute conflict as bitsets? Python ints are arbitrary precision.
    # N=7227. Bitset is 7227 bits ~ 904 bytes.
    # 7227 * 904 bytes ~ 6.5 MB. Very fast bitwise ops.
    
    # Convert conflict matrix to Python int bitsets
    # This is a structural change in representation.
    
    # But converting N x N bool matrix to N Python ints takes time.
    # Let's stick to numpy bool arrays for now, but optimize the check.
    
    # Actually, let's try the Add/Remove/SA approach.
    
    # 1. Greedy Init (already done)
    
    # 2. SA Loop
    t = T0
    steps = 0
    
    while time.time() < deadline - 5:
        # Choose move type
        # 60% Add, 20% Remove, 20% Swap?
        # Or just try Add first. If stuck, try Remove.
        
        move_type = rng.random()
        
        if move_type < 0.5:
            # Try Add
            unsel_idx = np.where(~selected)[0]
            if len(unsel_idx) == 0:
                continue
            v = unsel_idx[rng.integers(len(unsel_idx))]
            
            # Check if v conflicts with any selected
            # conflict[v, :] & selected
            # np.any(conflict[v, :] & selected)
            if not np.any(conflict[v, :] & selected):
                # Valid add
                selected[v] = True
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_selected = selected.copy()
                # Reset temperature? Or just continue.
                # Adding is always good.
            else:
                # Invalid add. Do nothing? Or accept with low prob?
                # Standard SA: reject.
                pass
                
        elif move_type < 0.8:
            # Try Remove
            sel_idx = np.where(selected)[0]
            if len(sel_idx) == 0:
                continue
            v = sel_idx[rng.integers(len(sel_idx))]
            
            # Calculate acceptance probability
            # Delta E = -1 (size decreases)
            # Accept with prob exp(-1 / t)
            if rng.random() < np.exp(-1.0 / t):
                selected[v] = False
                current_size -= 1
            # Else reject
            
        else:
            # Try Swap
            sel_idx = np.where(selected)[0]
            unsel_idx = np.where(~selected)[0]
            if len(sel_idx) == 0 or len(unsel_idx) == 0:
                continue
            
            i = sel_idx[rng.integers(len(sel_idx))]
            j = unsel_idx[rng.integers(len(unsel_idx))]
            
            # Check if swap is valid
            # Remove i, Add j.
            # j must not conflict with any (selected \ {i})
            
            is_conflict_with_i = conflict[j, i]
            num_conflicts_total = np.sum(conflict[j, :] & selected)
            
            if is_conflict_with_i:
                num_conflicts_after_removal = num_conflicts_total - 1
            else:
                num_conflicts_after_removal = num_conflicts_total
            
            if num_conflicts_after_removal == 0:
                # Valid swap
                selected[i] = False
                selected[j] = True
                # Size unchanged
                # Always accept valid swaps to explore
            else:
                # Invalid swap. Reject.
                pass
                
        steps += 1
        
        # Cooling
        # Cool every 100 steps?
        if steps % 100 == 0:
            t *= 0.99
            if t < 0.1:
                t = 0.1
                # If we hit min temp, we might want to restart or break?
                # Let's keep going with min temp for local search.
                # Or break if we've been at min temp for a while?
                # For now, just clamp.
        
        if time.time() > deadline - 2:
            break

    # Final validation
    if not np.any(best_selected):
        return candidates[:1]
        
    return np.array(candidates[best_selected], dtype=np.int64)

# EVOLVE-BLOCK-END
