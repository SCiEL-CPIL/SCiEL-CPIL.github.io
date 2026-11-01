import time
import itertools
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin before 120s kill

    # 1. Precomputation
    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))
    
    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)
    
    # Map t-subsets to indices
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Build sparse coverage: for each block, list of t-subset indices it covers
    block_to_tsubs = []
    for i, B in enumerate(all_kblocks):
        covered = []
        for ts in itertools.combinations(B, t):
            j = tsub_to_idx[ts]
            covered.append(j)
        block_to_tsubs.append(covered)
    
    # Build inverse: for each t-subset, list of block indices covering it
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    for b_idx, tsubs in enumerate(block_to_tsubs):
        for s_idx in tsubs:
            tsub_to_blocks[s_idx].append(b_idx)
    
    # Convert to numpy arrays for fast operations
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
    
    # 2. Greedy Initialization with sparse updates
    uncovered = np.ones(n_tsubs, dtype=bool)
    selected_indices = []
    
    # Initial gains: all blocks cover C(k,t) t-subsets
    k_choose_t = len(block_to_tsubs[0])
    gains = np.full(n_blocks, k_choose_t, dtype=np.int32)
    
    while np.any(uncovered):
        if time.time() > deadline - 60.0:
            break
        
        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break
        
        selected_indices.append(best_idx)
        
        # Update uncovered and gains sparsely
        for s_idx in block_to_tsubs[best_idx]:
            if uncovered[s_idx]:
                uncovered[s_idx] = False
                # Decrease gain for all blocks covering this t-subset
                gains[tsub_to_blocks_np[s_idx]] -= 1
    
    # 3. Local Optimization with Criticality-Based Swaps
    current_blocks = list(selected_indices)
    
    # Build cover counts
    cover_counts = np.zeros(n_tsubs, dtype=np.int32)
    for idx in current_blocks:
        for s in block_to_tsubs[idx]:
            cover_counts[s] += 1
    
    best_blocks = list(current_blocks)
    best_size = len(best_blocks)
    
    def reduce_redundant(blocks, counts):
        """Remove blocks that don't cover any unique t-subset."""
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for idx in blocks:
                # Check if any t-subset covered by idx has count == 1
                unique = False
                for s in block_to_tsubs[idx]:
                    if counts[s] == 1:
                        unique = True
                        break
                if unique:
                    new_blocks.append(idx)
                else:
                    for s in block_to_tsubs[idx]:
                        counts[s] -= 1
                    changed = True
            blocks = new_blocks
        return blocks
    
    def try_swap_remove(blocks, counts, block_to_remove):
        """
        Try to remove block_to_remove and repair with minimal additions.
        Returns (success, new_blocks, new_counts) or (False, None, None).
        """
        # Find t-subsets uniquely covered by block_to_remove
        unique_tsubs = []
        for s in block_to_tsubs[block_to_remove]:
            if counts[s] == 1:
                unique_tsubs.append(s)
        
        if not unique_tsubs:
            # Block is redundant, just remove it
            new_counts = counts.copy()
            for s in block_to_tsubs[block_to_remove]:
                new_counts[s] -= 1
            new_blocks = [b for b in blocks if b != block_to_remove]
            return True, new_blocks, new_counts
        
        # Need to find replacement blocks that cover all unique_tsubs
        # Start with candidates from the first unique t-subset
        first_s = unique_tsubs[0]
        candidates = tsub_to_blocks_np[first_s]
        
        # Filter candidates that are not already in blocks
        block_set = set(blocks)
        valid_candidates = [c for c in candidates if c not in block_set]
        
        if not valid_candidates:
            return False, None, None
        
        # Try each candidate (or a random subset for speed)
        random.shuffle(valid_candidates)
        max_candidates_to_try = min(len(valid_candidates), 50)
        
        for cand in valid_candidates[:max_candidates_to_try]:
            # Check if cand covers all unique_tsubs
            cand_tsub_set = set(block_to_tsubs[cand])
            if all(s in cand_tsub_set for s in unique_tsubs):
                # Found a single replacement
                new_counts = counts.copy()
                # Remove old block
                for s in block_to_tsubs[block_to_remove]:
                    new_counts[s] -= 1
                # Add new block
                for s in block_to_tsubs[cand]:
                    new_counts[s] += 1
                
                # Verify no t-subset becomes uncovered
                if np.any(new_counts < 0):
                    continue
                
                new_blocks = [b for b in blocks if b != block_to_remove]
                new_blocks.append(cand)
                
                # Check if this is an improvement (same size or better)
                # Actually, we're replacing 1 with 1, so size stays same
                # But we might enable further reductions
                return True, new_blocks, new_counts
        
        # Try 2-block replacement: find two blocks that together cover all unique_tsubs
        if len(unique_tsubs) <= 4:  # Only try for small sets
            for i in range(min(len(valid_candidates), 20)):
                for j in range(i+1, min(len(valid_candidates), 20)):
                    cand1, cand2 = valid_candidates[i], valid_candidates[j]
                    set1 = set(block_to_tsubs[cand1])
                    set2 = set(block_to_tsubs[cand2])
                    if all(s in set1 or s in set2 for s in unique_tsubs):
                        new_counts = counts.copy()
                        for s in block_to_tsubs[block_to_remove]:
                            new_counts[s] -= 1
                        for s in block_to_tsubs[cand1]:
                            new_counts[s] += 1
                        for s in block_to_tsubs[cand2]:
                            new_counts[s] += 1
                        
                        if np.any(new_counts < 0):
                            continue
                        
                        new_blocks = [b for b in blocks if b != block_to_remove]
                        new_blocks.extend([cand1, cand2])
                        return True, new_blocks, new_counts
        
        return False, None, None
    
    # Main optimization loop
    random.seed(42)
    no_improvement_count = 0
    
    def repair_greedy(blocks, counts):
        """Add blocks greedily to cover any uncovered t-subsets."""
        uncovered = np.where(counts == 0)[0]
        if len(uncovered) == 0:
            return blocks, counts
        
        # Build a set of current blocks for fast lookup
        block_set = set(blocks)
        
        # Calculate gain for each block: number of currently uncovered t-subsets it covers
        gains = np.zeros(n_blocks, dtype=np.int32)
        for s in uncovered:
            gains[tsub_to_blocks_np[s]] += 1
        
        while len(uncovered) > 0:
            if time.time() > deadline - 15.0:
                break
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[random.randint(0, len(candidates)-1)]
            
            if best_idx in block_set:
                # Should not happen if we update gains correctly, but safety
                gains[best_idx] = -1
                continue
                
            blocks.append(best_idx)
            block_set.add(best_idx)
            
            # Update counts and gains
            for s in block_to_tsubs[best_idx]:
                if counts[s] == 0:
                    counts[s] = 1
                    # This t-subset is now covered, so it no longer contributes to gains
                    # But we are using a static 'uncovered' list? No, we need to update uncovered.
                    # For efficiency in repair, we can just recompute uncovered if it's small.
                    pass
            
            # Recompute uncovered and gains for next iteration if set is small
            # Or just update counts and check if all covered
            # For speed, let's just update counts and rely on the outer loop to check validity?
            # No, repair must ensure validity.
            
            # Efficient update:
            # We added best_idx.
            # Any t-subset in best_idx that was 0 is now 1.
            # Gains for all blocks covering those t-subsets should decrease?
            # Actually, gains were calculated for the *initial* set of uncovered.
            # As we add blocks, the set of uncovered shrinks.
            # We should re-calculate gains for the remaining uncovered.
            
            remaining_uncovered = np.where(counts == 0)[0]
            if len(remaining_uncovered) == 0:
                break
            
            gains = np.zeros(n_blocks, dtype=np.int32)
            for s in remaining_uncovered:
                gains[tsub_to_blocks_np[s]] += 1
        
        return blocks, counts

    while time.time() < deadline - 10.0:
        # Step 1: Reduce redundant blocks
        current_blocks = reduce_redundant(current_blocks, cover_counts)
        
        if len(current_blocks) < best_size:
            best_blocks = list(current_blocks)
            best_size = len(current_blocks)
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        if no_improvement_count > 50:
            # Restart from best with different perturbation
            current_blocks = list(best_blocks)
            cover_counts = np.zeros(n_tsubs, dtype=np.int32)
            for idx in current_blocks:
                for s in block_to_tsubs[idx]:
                    cover_counts[s] += 1
            no_improvement_count = 0
        
        # Step 2: Try criticality-based swaps
        # Pick a random block to try removing
        if len(current_blocks) > 1:
            random.shuffle(current_blocks)
            for block_to_try in current_blocks[:min(5, len(current_blocks))]:
                success, new_blocks, new_counts = try_swap_remove(
                    current_blocks, cover_counts, block_to_try
                )
                if success:
                    # Check if we can further reduce
                    new_blocks = reduce_redundant(new_blocks, new_counts)
                    
                    if len(new_blocks) < len(current_blocks):
                        current_blocks = new_blocks
                        cover_counts = new_counts
                        if len(current_blocks) < best_size:
                            best_blocks = list(current_blocks)
                            best_size = len(current_blocks)
                            no_improvement_count = 0
                        break
                    elif len(new_blocks) == len(current_blocks):
                        # Same size, but might enable future reductions
                        # Accept with some probability
                        if random.random() < 0.3:
                            current_blocks = new_blocks
                            cover_counts = new_counts
        
        # Step 3: Kick - Remove a random block and repair (Packing-First style perturbation)
        if time.time() < deadline - 12.0 and len(current_blocks) > 10:
            # Pick a random block to remove
            kick_idx = random.choice(current_blocks)
            
            # Remove it
            temp_blocks = [b for b in current_blocks if b != kick_idx]
            temp_counts = cover_counts.copy()
            for s in block_to_tsubs[kick_idx]:
                temp_counts[s] -= 1
            
            # Repair the holes
            repaired_blocks, repaired_counts = repair_greedy(temp_blocks, temp_counts)
            
            # Check if valid and better
            if np.all(repaired_counts > 0):
                # Reduce redundancy in the repaired set
                repaired_blocks = reduce_redundant(repaired_blocks, repaired_counts)
                
                if len(repaired_blocks) < len(current_blocks):
                    current_blocks = repaired_blocks
                    cover_counts = repaired_counts
                    if len(current_blocks) < best_size:
                        best_blocks = list(current_blocks)
                        best_size = len(current_blocks)
                        no_improvement_count = 0
    
    # Final polish: aggressive redundancy removal and single-block swaps
    if time.time() < deadline - 5.0:
        final_blocks = list(best_blocks)
        final_counts = np.zeros(n_tsubs, dtype=np.int32)
        for idx in final_blocks:
            for s in block_to_tsubs[idx]:
                final_counts[s] += 1
        
        # Multiple passes of redundancy removal
        for _ in range(3):
            final_blocks = reduce_redundant(final_blocks, final_counts)
        
        # Try single-block removals one more time
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for block_to_try in final_blocks[:]:
                success, new_blocks, new_counts = try_swap_remove(
                    final_blocks, final_counts, block_to_try
                )
                if success and len(new_blocks) < len(final_blocks):
                    final_blocks = new_blocks
                    final_counts = new_counts
                    improved = True
                    break
    
    # Use the best of best_blocks and final_blocks
    if len(final_blocks) < len(best_blocks):
        best_blocks = final_blocks
    
    # Convert to actual blocks
    return [all_kblocks[i] for i in best_blocks]

if __name__ == "__main__":
    v, k, t = 19, 7, 4
    t0 = time.time()
    result = construct(v, k, t)
    t1 = time.time()
    print(f"C({v},{k},{t}): {len(result)} blocks, time {t1 - t0:.1f}s")
    # Verify
    all_tsubs = set(itertools.combinations(range(v), t))
    covered = set()
    for b in result:
        assert len(b) == k and len(set(b)) == k
        covered.update(itertools.combinations(b, t))
    assert covered == all_tsubs, "Invalid covering!"
    print("Valid!")
