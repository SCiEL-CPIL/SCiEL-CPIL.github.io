"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit
    
    N = 1000
    max_sum = 2 * N
    
    # State variables
    # mask[i] is True if i is in the set
    mask = np.zeros(N + 1, dtype=bool)
    # sum_counts[s] is the number of pairs (a,b) with a<=b, a,b in set, a+b=s
    sum_counts = np.zeros(max_sum + 1, dtype=np.int32)
    
    # Helper to add element x
    def add_element(x):
        if mask[x]:
            return
        # Update sum counts
        # x + x
        sum_counts[2 * x] += 1
        # x + y for all y in set
        if mask.any():
            idxs = np.nonzero(mask)[0]
            # Exclude x itself from the loop? No, x is not in mask yet.
            # So all y in mask are distinct from x.
            sums = idxs + x
            sum_counts[sums] += 1
        mask[x] = True

    # Helper to remove element x
    def remove_element(x):
        if not mask[x]:
            return
        # Update sum counts
        sum_counts[2 * x] -= 1
        if mask.any():
            idxs = np.nonzero(mask)[0]
            # Exclude x from idxs? Yes, because x is still in mask.
            # We need to subtract counts for pairs (x, y) where y != x.
            # The pair (x, x) was counted in 2*x.
            # The pairs (x, y) for y != x are in x+y.
            # So we subtract 1 from sum_counts[x+y] for all y in mask, y != x.
            other_idxs = idxs[idxs != x]
            if len(other_idxs) > 0:
                sums = other_idxs + x
                sum_counts[sums] -= 1
        mask[x] = False

    # Helper to check if x can be added
    def can_add(x):
        if mask[x]:
            return False
        # Check 2*x
        if sum_counts[2 * x] >= 2:
            return False
        # Check x + y for all y in set
        if mask.any():
            idxs = np.nonzero(mask)[0]
            sums = idxs + x
            # If any sum has count >= 2, adding x would make it 3
            if np.any(sum_counts[sums] >= 2):
                return False
        return True

    # Helper to get current size
    def get_size():
        return int(np.sum(mask))

    # Phase 1: Greedy Construction
    # Use a randomized order to diversify, but standard ascending is often good for density.
    # Let's try a few different greedy strategies to find a good start.
    
    best_mask = mask.copy()
    best_size = 0
    best_counts = sum_counts.copy()
    
    rng = random.Random(42)
    
    # Strategy 1: Ascending Greedy
    for x in range(1, N + 1):
        if can_add(x):
            add_element(x)
    size1 = get_size()
    if size1 > best_size:
        best_size = size1
        best_mask = mask.copy()
        best_counts = sum_counts.copy()
        
    # Reset state
    mask[:] = False
    sum_counts[:] = 0
    
    # Strategy 2: Descending Greedy
    for x in range(N, 0, -1):
        if can_add(x):
            add_element(x)
    size2 = get_size()
    if size2 > best_size:
        best_size = size2
        best_mask = mask.copy()
        best_counts = sum_counts.copy()
        
    # Reset state
    mask[:] = False
    sum_counts[:] = 0

    # Strategy 3: Randomized Greedy (few restarts)
    for restart in range(5):
        order = list(range(1, N + 1))
        rng.shuffle(order)
        for x in order:
            if can_add(x):
                add_element(x)
        size_r = get_size()
        if size_r > best_size:
            best_size = size_r
            best_mask = mask.copy()
            best_counts = sum_counts.copy()
        mask[:] = False
        sum_counts[:] = 0

    # Initialize current state with the best found so far
    mask[:] = best_mask
    sum_counts[:] = best_counts
    current_size = best_size
    
    # Phase 2: Local Search
    # We will perform moves to try to increase size.
    # Moves:
    # 1. Insert: Add a valid element. (Size +1)
    # 2. Swap: Remove one, add one. (Size same, changes configuration)
    # 3. Double Swap: Remove one, add two. (Size +1) - This is the most valuable.
    
    # To make "Double Swap" efficient, we can just try random pairs.
    # But first, let's do a pass of simple Inserts if any exist (should be none in greedy, but maybe after swaps).
    
    in_set_list = np.nonzero(mask)[0]
    not_in_set_list = np.nonzero(~mask)[0]
    
    # We need to maintain these lists or regenerate them periodically.
    # Regenerating is O(N) which is fast.
    
    time_budget = deadline - time.time()
    if time_budget < 1.0:
        return list(in_set_list)

    # Simulated Annealing / Random Walk on valid states
    # We accept neutral moves (size same) to explore the state space.
    # We accept improving moves (size +1) always.
    
    # To speed up, we pre-compute "conflict" info? 
    # Actually, can_add is O(|S|) with NumPy vectorization, which is very fast.
    
    # Let's iterate
    step_count = 0
    while time.time() < deadline - 0.5:
        step_count += 1
        
        # 1. Try Insert
        # Check if any element in not_in_set can be added.
        # This is expensive if we check all. 
        # Instead, pick a random candidate.
        if len(not_in_set_list) > 0:
            # Sample a few candidates to see if any are insertable
            # If we find one, add it.
            candidates_to_try = rng.sample(not_in_set_list, min(10, len(not_in_set_list)))
            inserted = False
            for x in candidates_to_try:
                if can_add(x):
                    add_element(x)
                    current_size += 1
                    inserted = True
                    # Update lists
                    in_set_list = np.nonzero(mask)[0]
                    not_in_set_list = np.nonzero(~mask)[0]
                    break
            
            if inserted:
                continue # Restart loop with new state
        
        # 2. Try Swap (Neutral Move)
        # Remove a random u, try to add a random v.
        if len(in_set_list) > 0 and len(not_in_set_list) > 0:
            u = int(rng.choice(in_set_list))
            v = int(rng.choice(not_in_set_list))
            
            # Try swap
            remove_element(u)
            
            if can_add(v):
                add_element(v)
                # Size is same. Accept this move.
                in_set_list = np.nonzero(mask)[0]
                not_in_set_list = np.nonzero(~mask)[0]
            else:
                # Revert
                remove_element(v) if mask[v] else None # v was not added
                # Wait, can_add(v) failed, so v was NOT added.
                # We need to re-add u.
                add_element(u)
                # State is restored.
        
        # 3. Try Double Swap (Remove 1, Add 2)
        # This is harder to do efficiently.
        # Remove u.
        # Then try to add v1 and v2.
        # If we remove u, we free up capacity in sums.
        # Let's try this occasionally.
        if len(in_set_list) > 0 and len(not_in_set_list) > 1:
            if rng.random() < 0.1: # 10% chance to try double swap
                u = int(rng.choice(in_set_list))
                remove_element(u)
                
                # Now try to add two elements
                # Pick two random candidates
                # We must ensure they are distinct and not in set
                # Note: u is not in set now.
                # not_in_set_list is stale, so regenerate or pick from range
                # Picking from range is safer but slower. 
                # Let's pick from the global range 1..N excluding current set.
                # Since N=1000, we can just pick random integers.
                
                added_any = False
                added_v1 = -1
                added_v2 = -1
                
                # Try to find a valid v1
                for _ in range(5): # Try 5 times
                    v1 = rng.randint(1, N)
                    if not mask[v1] and can_add(v1):
                        add_element(v1)
                        added_v1 = v1
                        added_any = True
                        break
                
                if added_any:
                    # Try to find a valid v2
                    for _ in range(5):
                        v2 = rng.randint(1, N)
                        if v2 != added_v1 and not mask[v2] and can_add(v2):
                            add_element(v2)
                            added_v2 = v2
                            break
                
                if added_v1 != -1 and added_v2 != -1:
                    # Success! Size increased by 1 (removed 1, added 2)
                    current_size += 1
                    in_set_list = np.nonzero(mask)[0]
                    not_in_set_list = np.nonzero(~mask)[0]
                else:
                    # Failed to add 2. 
                    # If we added 1, we are at same size as before (removed 1, added 1).
                    # If we added 0, we are at size - 1 (invalid, must restore).
                    # We must restore u if we didn't end up with a valid set of size >= original.
                    # Actually, if we added 1, size is same. We can accept or revert.
                    # If we added 0, size is smaller. We MUST revert to keep size.
                    
                    if added_v1 == -1:
                        # Nothing added, u is still out. Re-add u.
                        add_element(u)
                    else:
                        # Added 1. Size is same. 
                        # Should we keep this configuration? 
                        # It might help future double swaps. 
                        # Let's keep it (neutral move).
                        # But we lost u. 
                        # in_set_list needs update.
                        in_set_list = np.nonzero(mask)[0]
                        not_in_set_list = np.nonzero(~mask)[0]
                        # Note: current_size is unchanged.
                
    # Final Result
    # Ensure we return distinct integers
    result = np.nonzero(mask)[0]
    return list(result)

# EVOLVE-BLOCK-END
