import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    DEADLINE = start_time + 125.0  # 5s safety margin

    points = list(range(v))

    # 1. Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)

    # 2. Generate all t-subsets and map to indices
    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}

    # 3. Precompute block -> t-subsets mapping
    block_subs = []
    for b in all_blocks:
        subs = []
        for s in itertools.combinations(b, t):
            subs.append(t_idx[s])
        block_subs.append(subs)
    
    # Convert to numpy arrays for faster access if possible, but lists are fine
    # Let's keep lists for flexibility in iteration
    
    # 4. Precompute t-subset -> blocks mapping
    sub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_subs):
        for si in subs:
            sub_blocks[si].append(i)

    def run_greedy(random_seed=None, noise_scale=0.0, alpha=1.0):
        """
        Greedy set cover with randomization.
        alpha controls the balance between gain and noise (if we use a soft-max style)
        Here we just add noise to gain.
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # gain[i] = number of currently uncovered t-subsets covered by block i
        gain = np.array([len(subs) for subs in block_subs], dtype=np.int32)
        is_covered = np.zeros(N_t, dtype=bool)
        uncovered_count = N_t
        selected_blocks = []
        
        # Precompute the list of blocks for each subset for fast gain update
        # We already have sub_blocks
        
        while uncovered_count > 0 and time.time() < DEADLINE - 5.0:
            # Find best block
            if noise_scale > 0:
                noise = np.random.exponential(noise_scale, size=N_blocks)
                scores = gain + noise
            else:
                scores = gain
                
            best_idx = np.argmax(scores)
            
            if gain[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            gain[best_idx] = -1  # Disable this block
            
            # Update gains for all blocks that share subsets with best_idx
            for si in block_subs[best_idx]:
                if not is_covered[si]:
                    is_covered[si] = True
                    uncovered_count -= 1
                    for b_idx in sub_blocks[si]:
                        if gain[b_idx] > 0:
                            gain[b_idx] -= 1
                            
        return selected_blocks

    def reduce_solution(block_indices):
        """
        Remove redundant blocks.
        """
        if not block_indices:
            return block_indices
            
        # Count coverage for each t-subset
        cov_count = np.zeros(N_t, dtype=np.int32)
        for b_idx in block_indices:
            for si in block_subs[b_idx]:
                cov_count[si] += 1
                
        changed = True
        current_blocks = list(block_indices)
        
        while changed and time.time() < DEADLINE - 5.0:
            changed = False
            new_blocks = []
            for b_idx in current_blocks:
                redundant = True
                for si in block_subs[b_idx]:
                    if cov_count[si] <= 1:
                        redundant = False
                        break
                if redundant:
                    changed = True
                    for si in block_subs[b_idx]:
                        cov_count[si] -= 1
                else:
                    new_blocks.append(b_idx)
            if changed:
                current_blocks = new_blocks
                
        return current_blocks

    def local_search(block_indices):
        """
        Local search via swaps and removals.
        1. Try to remove any block (if redundant).
        2. Try to swap a block with another block that covers its unique subsets,
           potentially enabling further removals.
        """
        if not block_indices:
            return block_indices
            
        current_blocks = list(block_indices)
        block_set = set(current_blocks)
        
        # Count coverage
        cov_count = np.zeros(N_t, dtype=np.int32)
        for b_idx in current_blocks:
            for si in block_subs[b_idx]:
                cov_count[si] += 1
                
        improved = True
        passes = 0
        max_passes = 50
        
        while improved and time.time() < DEADLINE - 10.0 and passes < max_passes:
            improved = False
            passes += 1
            new_blocks = []
            
            # Shuffle to avoid bias
            idx_list = list(range(len(current_blocks)))
            random.shuffle(idx_list)
            
            for pos in idx_list:
                b_idx = current_blocks[pos]
                
                # Check redundancy
                redundant = True
                for si in block_subs[b_idx]:
                    if cov_count[si] <= 1:
                        redundant = False
                        break
                        
                if redundant:
                    # Remove it
                    for si in block_subs[b_idx]:
                        cov_count[si] -= 1
                    block_set.discard(b_idx)
                    improved = True
                    continue
                
                # Try to find a replacement
                # Critical subsets: those covered ONLY by b_idx
                critical_subs = []
                for si in block_subs[b_idx]:
                    if cov_count[si] == 1:
                        critical_subs.append(si)
                        
                if not critical_subs:
                    # Should be redundant, but just in case
                    new_blocks.append(b_idx)
                    continue
                    
                # Find candidates that cover all critical_subs
                # To speed up, we can pick a random critical subset and look at blocks covering it
                # Then check if they cover all critical_subs
                cand_subset = random.choice(critical_subs)
                candidates = sub_blocks[cand_subset]
                
                best_swap = -1
                best_gain = 0 # Gain in terms of "extra" coverage or enabling removals?
                
                # For each candidate, check if it covers all critical_subs
                # If yes, calculate "potential benefit":
                # The new block might cover subsets that are currently covered by other blocks,
                # making those other blocks redundant.
                
                # To keep it fast, we just check if a swap is valid (covers criticals)
                # and if the new block is "better" in some heuristic sense.
                # Heuristic: The new block should cover subsets that are heavily covered by others?
                # Or simply: try the swap, run reduction, see if size decreases.
                # This is expensive.
                
                # Faster heuristic:
                # Count how many subsets the new block covers that are currently covered by >= 2 blocks.
                # If we swap, we add 1 to their count. This doesn't directly help removal.
                # Removal helps if a block's unique subsets become covered by others.
                
                # Let's try a specific move:
                # If we replace b_idx with c_idx:
                # 1. b_idx is removed. Subsets covered only by b_idx must be covered by c_idx.
                # 2. c_idx is added.
                # 3. This might make some other block d_idx redundant if its unique subsets were covered by b_idx?
                #    No, b_idx is removed, so subsets covered by b_idx lose coverage.
                #    Wait, if c_idx covers them, they are still covered.
                #    But what about subsets covered by b_idx and d_idx?
                #    If b_idx is removed, those subsets lose one coverage.
                #    If they were covered by exactly 2 blocks (b_idx and d_idx), they are now covered by 1.
                #    This might make d_idx NOT redundant if it wasn't before?
                #    Actually, we want to find a swap that allows REMOVING another block.
                
                # Let's stick to a simpler, effective local search:
                # 1. Identify "critical" blocks (blocks with unique subsets).
                # 2. For a critical block b, find a replacement c that covers b's unique subsets.
                # 3. Perform the swap.
                # 4. Run a fast reduction pass.
                # 5. If the solution size didn't increase (it shouldn't, since we swapped 1 for 1) 
                #    and we were able to remove more blocks in the reduction pass, keep it.
                #    Otherwise, revert.
                
                # This is O(N_blocks) per check, which is too slow if done for every block.
                # We limit the number of swaps attempted per pass.
                
                # Optimization: Only try swaps for a few random blocks per pass.
                if random.random() < 0.1: # Only try swap for 10% of blocks
                    # Find candidates
                    valid_candidates = []
                    for c_idx in candidates:
                        if c_idx == b_idx or c_idx in block_set:
                            continue
                        # Check if c_idx covers all critical_subs
                        covers_all = True
                        for cs in critical_subs:
                            if cs not in set(block_subs[c_idx]): # This is slow
                                covers_all = False
                                break
                        if covers_all:
                            valid_candidates.append(c_idx)
                            
                    if valid_candidates:
                        c_idx = random.choice(valid_candidates)
                        
                        # Perform trial swap
                        # 1. Remove b_idx
                        for si in block_subs[b_idx]:
                            cov_count[si] -= 1
                        # 2. Add c_idx
                        for si in block_subs[c_idx]:
                            cov_count[si] += 1
                        
                        # 3. Try to reduce
                        # We can do a quick check: are any blocks now redundant?
                        # Let's just run the full reduction on the modified set?
                        # That's expensive.
                        
                        # Alternative: Just check if the swap itself allows a direct removal.
                        # If not, revert.
                        
                        # Let's try a different approach:
                        # Instead of complex swaps, let's use a "repair" based local search.
                        # Pick a random block, remove it.
                        # If solution is invalid, find a block that covers the missing subsets.
                        # Add it.
                        # Then reduce.
                        
                        # This is effectively what the swap above does, but simpler to implement safely.
                        
                        # Let's revert the trial swap for now and use the repair strategy in a separate function or later.
                        for si in block_subs[c_idx]:
                            cov_count[si] -= 1
                        for si in block_subs[b_idx]:
                            cov_count[si] += 1
                        
                new_blocks.append(b_idx)
                
            if improved:
                current_blocks = new_blocks
                # Note: cov_count is updated correctly during removals.
                # If we didn't remove anything, cov_count is unchanged.
                
        return current_blocks

    def repair_local_search(block_indices):
        """
        Repair-based local search.
        1. Pick a random block to remove.
        2. Identify uncovered subsets.
        3. Find a block that covers all uncovered subsets.
        4. If found, add it.
        5. Reduce.
        6. If size is smaller, keep; else revert.
        """
        if not block_indices:
            return block_indices
            
        current_blocks = list(block_indices)
        block_set = set(current_blocks)
        
        # Count coverage
        cov_count = np.zeros(N_t, dtype=np.int32)
        for b_idx in current_blocks:
            for si in block_subs[b_idx]:
                cov_count[si] += 1
                
        best_sol = current_blocks
        best_len = len(current_blocks)
        
        attempts = 0
        max_attempts = 100
        
        while time.time() < DEADLINE - 15.0 and attempts < max_attempts:
            attempts += 1
            
            # Pick a random block to remove
            if not current_blocks:
                break
                
            rem_idx = random.choice(range(len(current_blocks)))
            b_idx = current_blocks[rem_idx]
            
            # Remove it
            uncovered = []
            for si in block_subs[b_idx]:
                cov_count[si] -= 1
                if cov_count[si] == 0:
                    uncovered.append(si)
                    
            if not uncovered:
                # It was redundant, just remove it
                current_blocks.pop(rem_idx)
                block_set.discard(b_idx)
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_sol = list(current_blocks)
                continue
            
            # Find a block that covers all uncovered subsets
            # Use the first uncovered subset to filter candidates
            cand_subset = uncovered[0]
            candidates = sub_blocks[cand_subset]
            
            found = False
            for c_idx in candidates:
                if c_idx == b_idx or c_idx in block_set:
                    continue
                # Check if c_idx covers all uncovered
                valid = True
                for u in uncovered:
                    if u not in set(block_subs[c_idx]):
                        valid = False
                        break
                if valid:
                    # Found a replacement
                    # Add it
                    for si in block_subs[c_idx]:
                        cov_count[si] += 1
                    current_blocks[rem_idx] = c_idx
                    block_set.discard(b_idx)
                    block_set.add(c_idx)
                    found = True
                    
                    # Now try to reduce the new solution
                    new_blocks = reduce_solution(current_blocks)
                    
                    if len(new_blocks) < best_len:
                        best_len = len(new_blocks)
                        best_sol = list(new_blocks)
                        # Update current state to the reduced one
                        current_blocks = new_blocks
                        block_set = set(current_blocks)
                        # Recalculate cov_count for current_blocks
                        cov_count = np.zeros(N_t, dtype=np.int32)
                        for b_idx in current_blocks:
                            for si in block_subs[b_idx]:
                                cov_count[si] += 1
                    else:
                        # Revert
                        for si in block_subs[c_idx]:
                            cov_count[si] -= 1
                        for si in block_subs[b_idx]:
                            cov_count[si] += 1
                        current_blocks[rem_idx] = b_idx
                        block_set.discard(c_idx)
                        block_set.add(b_idx)
                    break # Only try one candidate per removal attempt? Or try more?
                    # Let's break to keep it fast.
                    
            if not found:
                # Revert
                for si in block_subs[b_idx]:
                    cov_count[si] += 1
                    
        return best_sol

    best_sol = None
    best_len = float('inf')
    
    # Run deterministic greedy first
    sol = run_greedy(random_seed=None)
    sol = reduce_solution(sol)
    best_sol = sol
    best_len = len(sol)
    
    # Run randomized restarts
    attempt = 0
    while time.time() < DEADLINE - 15.0:
        # Vary parameters
        noise_scale = 0.05 * (1 + attempt % 5)
        sol = run_greedy(random_seed=attempt * 13 + 7, noise_scale=noise_scale)
        
        if not sol:
            attempt += 1
            continue
            
        sol = reduce_solution(sol)
        
        # Local search
        if time.time() < DEADLINE - 20.0:
            sol = local_search(sol)
            sol = reduce_solution(sol)
            
        # Repair local search
        if time.time() < DEADLINE - 15.0:
            sol = repair_local_search(sol)
            sol = reduce_solution(sol)
            
        if len(sol) < best_len:
            best_len = len(sol)
            best_sol = sol
            
        attempt += 1
        
    # Final reduction
    if best_sol:
        best_sol = reduce_solution(best_sol)
        
    return [all_blocks[i] for i in best_sol]
