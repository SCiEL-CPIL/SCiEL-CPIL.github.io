import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generate all D_11 vectors with squared norm 4
    candidates = []
    
    # Type 1: One ±2, rest 0
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four ±1's, rest 0
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # Conflict: two vectors are incompatible if M[i,j] > 2
    conflict = (M > 2).astype(np.uint8)
    np.fill_diagonal(conflict, 0)
    
    rng = np.random.default_rng(42)
    
    # Initial Greedy: Sort by degree (fewest conflicts first is often better for Independent Set? 
    # Actually, for Independent Set, we want nodes that don't conflict. 
    # "Greedy by degree" usually means pick node with min degree first to leave more options? 
    # Or max degree to "dominate" the graph? 
    # For MIS, picking low degree nodes first is a common heuristic.
    degrees = conflict.sum(axis=1)
    
    best_set = []
    best_size = 0
    
    # Try a few different greedy orderings
    orderings = [
        np.argsort(degrees),       # Min degree first
        np.argsort(-degrees),      # Max degree first
        rng.permutation(N),
        rng.permutation(N),
        rng.permutation(N)
    ]
    
    for order in orderings:
        if time.time() > deadline - 50:
            break
        in_set = np.zeros(N, dtype=bool)
        conflict_counts = np.zeros(N, dtype=np.int32)
        
        for idx in order:
            if conflict_counts[idx] == 0:
                in_set[idx] = True
                conflict_counts += conflict[idx]
        
        current_size = np.sum(in_set)
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
            best_conflict_counts = conflict_counts.copy()
    
    # Local Search: Batch Swaps
    # We use the best initial set
    in_set = best_in_set.copy()
    conflict_counts = best_conflict_counts.copy()
    current_size = best_size
    
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        improved = False
        
        # 1. Try to add any node with 0 conflicts
        addable = np.where((~in_set) & (conflict_counts == 0))[0]
        if len(addable) > 0:
            # Add all of them? Or just one? Adding all is safe if they don't conflict with each other.
            # But they might conflict with each other.
            # To be safe, add one by one, but vectorize the check if possible.
            # For speed, let's just add the first one found, or iterate.
            # Actually, if conflict_counts[i] == 0, it has no conflict with CURRENT set.
            # It might have conflict with OTHER addable nodes.
            # Let's just pick one randomly to keep it simple and fast.
            idx = addable[rng.integers(len(addable))]
            in_set[idx] = True
            conflict_counts += conflict[idx]
            current_size += 1
            improved = True
            no_improve = 0
            continue
            
        # 2. Try Batch Swap: Remove one node i, add multiple nodes j that were blocked only by i
        # Identify nodes j not in set that have exactly 1 conflict
        # conflict_counts[j] == 1
        # The single conflict must be in the set.
        # Let's find for each j in set, how many "blocked by 1" nodes it blocks.
        
        # Get indices of nodes with exactly 1 conflict
        blocked_by_one = np.where((~in_set) & (conflict_counts == 1))[0]
        
        if len(blocked_by_one) == 0:
            # Try random single swap (remove i, add j)
            # Find j with min conflicts?
            # Or just break if no easy moves
            # Let's try a random swap: pick random i in set, random j not in set.
            # If removing i makes j addable, do it.
            selected_idx = np.where(in_set)[0]
            unselected_idx = np.where(~in_set)[0]
            
            if len(selected_idx) == 0 or len(unselected_idx) == 0:
                break
                
            # Pick random i
            i = selected_idx[rng.integers(len(selected_idx))]
            
            # Which unselected nodes become addable if i is removed?
            # A node j becomes addable if conflict_counts[j] == 1 AND conflict[i, j] == 1
            # OR conflict_counts[j] == 0 (handled above)
            
            # Check all j
            # potential_j = unselected_idx
            # valid_j = potential_j[conflict[i, potential_j] == 1]
            # valid_j = valid_j[conflict_counts[valid_j] == 1]
            
            # To be faster, we only care if we can add MORE than 1? No, even 1 is an improvement if we can keep size? 
            # Wait, removing 1 and adding 1 keeps size same. We want to INCREASE size.
            # So we need to remove 1 and add > 1.
            # Or remove 0 and add 1 (handled above).
            
            if len(blocked_by_one) > 0:
                # For each i in set, count how many blocked_by_one nodes it conflicts with
                # conflict[i, blocked_by_one]
                # Sum over axis 1?
                # conflict is N x N.
                # We want counts for each i in set.
                
                # Mask for set
                in_set_mask = in_set
                # Get conflict matrix rows for set nodes
                # This is expensive if done every iteration.
                # Optimization: conflict is sparse-ish? No, dense boolean.
                
                # Let's just do it if N is small enough or time allows.
                # N ~ 5000. 5000x5000 matrix.
                # conflict[in_set_mask, :].sum(axis=1) -> gives count of conflicts for each i in set with ALL nodes.
                # We specifically want conflicts with blocked_by_one.
                
                # Create a mask for blocked_by_one
                blocked_mask = np.zeros(N, dtype=bool)
                blocked_mask[blocked_by_one] = True
                
                # Count conflicts between set nodes and blocked nodes
                # conflict_set_blocked = conflict[in_set_mask, :][:, blocked_mask].sum(axis=1)
                
                # This creates a large intermediate array.
                # Alternative: 
                # For each j in blocked_by_one, find its unique conflict i.
                # Since conflict_counts[j] == 1, there is exactly one i in set such that conflict[i,j]==1.
                # We can find this i by:
                # idx_i = np.where(conflict[:, j] & in_set)[0][0]
                
                # This is O(N) per j. If we have many j's, it's slow.
                # But we can vectorize?
                # Let's try a simpler heuristic:
                # Pick a random i from set.
                # Check how many blocked_by_one nodes it conflicts with.
                # If > 1, remove i, add them all.
                
                i = selected_idx[rng.integers(len(selected_idx))]
                
                # Find which blocked_by_one nodes conflict with i
                # conflict[i, blocked_by_one]
                conflicts_with_i = blocked_by_one[conflict[i, blocked_by_one] == 1]
                
                if len(conflicts_with_i) > 1:
                    # Remove i
                    in_set[i] = False
                    conflict_counts -= conflict[i]
                    current_size -= 1
                    
                    # Add all conflicts_with_i
                    # Check if they are still addable (they should be, as their only conflict was i)
                    # But they might conflict with EACH OTHER.
                    # We need to select an independent set from conflicts_with_i.
                    # Greedy:
                    addable_now = conflicts_with_i.copy()
                    added = 0
                    while len(addable_now) > 0:
                        # Pick one
                        j = addable_now[rng.integers(len(addable_now))]
                        if conflict_counts[j] == 0:
                            in_set[j] = True
                            conflict_counts += conflict[j]
                            current_size += 1
                            added += 1
                            # Remove j from addable_now? 
                            # Other nodes in addable_now might conflict with j.
                            # Update addable_now to remove those that now have conflict > 0?
                            # conflict_counts updated.
                            addable_now = addable_now[conflict_counts[addable_now] == 0]
                        else:
                            # Should not happen if logic is correct, but skip
                            addable_now = np.delete(addable_now, np.where(addable_now == j)[0][0])
                    
                    if current_size > best_size:
                        best_size = current_size
                        best_in_set = in_set.copy()
                        best_conflict_counts = conflict_counts.copy()
                        no_improve = 0
                        improved = True
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            else:
                no_improve += 1
        else:
            no_improve += 1

    # Finalize
    if best_size == 0:
        return candidates[:22].astype(np.int64)
        
    result = candidates[best_in_set].astype(np.int64)
    
    # Verify
    n = len(result)
    if n == 0:
        return candidates[:22].astype(np.int64)
        
    max_norm_sq = int(np.max(np.sum(result**2, axis=1)))
    diffs = result[:, np.newaxis, :] - result[np.newaxis, :, :]
    diff_norms_sq = np.sum(diffs**2, axis=2)
    if np.min(diff_norms_sq[np.triu_indices(n, k=1)]) < max_norm_sq:
        return candidates[:22].astype(np.int64)
        
    return result
