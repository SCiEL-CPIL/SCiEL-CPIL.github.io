import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    # 1. Precomputation
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    t_tuple_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    num_t = len(t_subsets)
    
    # Generate all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)
    
    # Map block index to list of t-subset indices it covers
    block_to_ts = []
    # Map t-subset index to list of block indices that cover it
    ts_to_block = [[] for _ in range(num_t)]
    
    for j, blk in enumerate(k_blocks):
        ts_list = []
        for ts in itertools.combinations(blk, t):
            idx = t_tuple_to_idx[ts]
            ts_list.append(idx)
            ts_to_block[idx].append(j)
        block_to_ts.append(ts_list)
        
    # 2. Greedy Initialization
    def run_greedy(seed):
        rng = random.Random(seed)
        gain = np.full(num_k, 0, dtype=np.int32)
        is_uncovered = np.full(num_t, True, dtype=bool)
        is_selected = np.full(num_k, False, dtype=bool)
        sol = []
        
        # Initial gain is C(k, t) for all blocks
        gain[:] = C(k, t)
        
        while True:
            if time.time() > deadline - 5.0:
                return None
            
            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break
            
            # Find candidates with max gain
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
            
            b_idx = int(candidates[rng.randint(0, len(candidates) - 1)])
            
            sol.append(b_idx)
            is_selected[b_idx] = True
            gain[b_idx] = -1
            
            bts = block_to_ts[b_idx]
            
            # Find newly covered t-subsets
            newly_covered = [ts_idx for ts_idx in bts if is_uncovered[ts_idx]]
            
            if not newly_covered:
                continue
                
            # Mark as covered
            for ts_idx in newly_covered:
                is_uncovered[ts_idx] = False
                
            # Update gains
            for ts_idx in newly_covered:
                for b in ts_to_block[ts_idx]:
                    if not is_selected[b]:
                        gain[b] -= 1
                        
        return sol

    # Helper to calculate binomial coefficient (simple for small numbers)
    def C(n, r):
        if r > n: return 0
        if r == 0: return 1
        result = 1
        for i in range(r):
            result = result * (n - i) // (i + 1)
        return result

    best_solution = None
    best_count = float('inf')
    
    # We will run several greedy starts and refine the best one
    num_restarts = 0
    while time.time() < deadline - 20.0:
        num_restarts += 1
        seed = num_restarts * 13
        sol_indices = run_greedy(seed)
        
        if sol_indices is None:
            continue
            
        current_count = len(sol_indices)
        if current_count < best_count:
            best_count = current_count
            best_solution = sol_indices[:]
            
        # If we have a good candidate, try to improve it with local search
        # We only do local search on the best candidate found so far to save time?
        # Or do it on every candidate if time permits?
        # Let's do it on the current candidate if it's promising or if we have time.
        
        if time.time() < deadline - 10.0:
            improved = local_search(sol_indices, deadline)
            if len(improved) < best_count:
                best_count = len(improved)
                best_solution = improved

    if best_solution is None:
        # Fallback
        return []
        
    # Convert indices back to blocks
    return [list(k_blocks[idx]) for idx in best_solution]

def local_search(initial_indices, deadline):
    """
    Perform local search on the set of block indices.
    Moves:
    1. Remove redundant blocks.
    2. Swap a block with another to enable removal.
    """
    # Work with a copy
    current = initial_indices[:]
    
    # Build coverage counts for current solution
    # cover_count[ts_idx] = number of blocks in 'current' covering ts_idx
    cover_count = np.zeros(len(ts_to_block), dtype=np.int32) # Use global ts_to_block size
    # Note: ts_to_block is global, size is num_t
    # We need to map global ts index to count.
    
    # To make this fast, we maintain a set of selected block indices for O(1) lookup
    selected_set = set(current)
    
    # Initialize counts
    for b_idx in current:
        for ts_idx in block_to_ts[b_idx]:
            cover_count[ts_idx] += 1
            
    best_count = len(current)
    best_state = current[:]
    
    # Local search loop
    # We iterate through passes. Each pass tries to remove or swap.
    # To avoid infinite loops, we limit swaps or use a randomization.
    
    max_passes = 20
    for pass_num in range(max_passes):
        if time.time() > deadline - 2.0:
            break
            
        changed = False
        
        # Step 1: Remove redundant blocks
        # We scan the current list. If a block is redundant, remove it.
        # We iterate in reverse to allow popping, or use a new list.
        # Using a new list is safer and fast enough for ~300 blocks.
        new_current = []
        for b_idx in current:
            # Check if b_idx is redundant
            redundant = True
            for ts_idx in block_to_ts[b_idx]:
                if cover_count[ts_idx] <= 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove it
                for ts_idx in block_to_ts[b_idx]:
                    cover_count[ts_idx] -= 1
                selected_set.discard(b_idx)
                changed = True
            else:
                new_current.append(b_idx)
                
        current = new_current
        
        if len(current) < best_count:
            best_count = len(current)
            best_state = current[:]
            
        # Step 2: Swap improvement
        # For each block B in current, see if we can swap it with B' such that
        # the new state allows for a reduction (immediate or potential).
        # Heuristic: Try to swap B with B' if B' covers the "essential" t-subsets of B.
        # Essential t-subsets of B are those with cover_count == 1.
        
        # To make this efficient, we don't try ALL swaps. We try a limited number of random swaps or specific candidates.
        # Given the time budget, we can try to swap each block once per pass if beneficial.
        
        swaps_made = 0
        # Shuffle to avoid bias? Or just iterate.
        # Let's iterate through current blocks.
        for i in range(len(current)):
            if time.time() > deadline - 2.0:
                break
                
            b_idx = current[i]
            
            # Find essential t-subsets
            essential_ts = []
            for ts_idx in block_to_ts[b_idx]:
                if cover_count[ts_idx] == 1:
                    essential_ts.append(ts_idx)
                    
            if not essential_ts:
                continue # Should have been removed in Step 1, but safety check
                
            # Find candidate blocks B' that cover all essential_ts
            # Candidate must be in intersection of ts_to_block[ts] for ts in essential_ts
            # Start with the list for the first essential ts
            candidates = ts_to_block[essential_ts[0]]
            
            # Filter candidates:
            # 1. Not in selected_set
            # 2. Covers all other essential_ts
            
            valid_candidates = []
            for cand in candidates:
                if cand in selected_set:
                    continue
                # Check if cand covers all essential_ts
                cand_ts_set = set(block_to_ts[cand])
                # Optimization: Instead of set, check loop
                covers_all = True
                for ts_idx in essential_ts:
                    if ts_idx not in cand_ts_set: # This is O(1) if we use a set, but block_to_ts is list.
                        covers_all = False
                        break
                if covers_all:
                    valid_candidates.append(cand)
                    
            if not valid_candidates:
                continue
                
            # Pick a random candidate or the first one?
            # Random is better for escaping local optima.
            cand = random.choice(valid_candidates)
            
            # Try the swap: Remove b_idx, Add cand
            # Check if this swap is "valid" (it is by construction for essential_ts)
            # But does it break coverage of other t-subsets?
            # No, because b_idx is removed, so its t-subsets lose 1 count.
            # The essential_ts were count 1, so they become 0 unless covered by cand.
            # We ensured cand covers essential_ts.
            # Other t-subsets of b_idx had count > 1, so they remain covered (count >= 1).
            # So the swap is always valid!
            
            # However, we only want to swap if it helps.
            # How to judge?
            # 1. Immediate: Does it allow removal of other blocks?
            # 2. Potential: Does it increase overlap (making future removals easier)?
            
            # Let's perform the swap and see if we can remove redundant blocks immediately.
            
            # Tentatively apply swap
            for ts_idx in block_to_ts[b_idx]:
                cover_count[ts_idx] -= 1
            selected_set.discard(b_idx)
            
            for ts_idx in block_to_ts[cand]:
                cover_count[ts_idx] += 1
            selected_set.add(cand)
            
            current[i] = cand
            
            # Try to remove redundant blocks resulting from this swap
            # We do a quick pass to see if size reduces
            new_current = []
            local_removed = 0
            for b in current:
                redundant = True
                for ts_idx in block_to_ts[b]:
                    if cover_count[ts_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    for ts_idx in block_to_ts[b]:
                        cover_count[ts_idx] -= 1
                    selected_set.discard(b)
                    local_removed += 1
                else:
                    new_current.append(b)
                    
            if local_removed > 0:
                current = new_current
                changed = True
                swaps_made += 1
                if len(current) < best_count:
                    best_count = len(current)
                    best_state = current[:]
            else:
                # Revert swap if it didn't lead to immediate improvement
                # This is a greedy hill climbing move.
                for ts_idx in block_to_ts[cand]:
                    cover_count[ts_idx] -= 1
                selected_set.discard(cand)
                
                for ts_idx in block_to_ts[b_idx]:
                    cover_count[ts_idx] += 1
                selected_set.add(b_idx)
                
                current[i] = b_idx

        if not changed:
            break
            
    return best_state
