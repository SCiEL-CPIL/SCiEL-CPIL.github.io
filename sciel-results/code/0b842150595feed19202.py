import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    """
    Finds a covering design C(v, k, t) using Simulated Annealing.
    v=19, k=7, t=4
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute all t-subsets and their indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute for each block, the list of t-subset indices it covers
    # block_tsubs[bi] is a numpy array of t-subset indices
    block_tsubs = []
    for block in all_blocks:
        tsubs = []
        for ts in itertools.combinations(block, t):
            tsubs.append(tsub_to_idx[ts])
        block_tsubs.append(np.array(tsubs, dtype=np.int32))
        
    # Precompute for each t-subset, the list of blocks that cover it
    tsub_to_blocks = [[] for _ in range(n_tsubsets)]
    for bi, tsubs in enumerate(block_tsubs):
        for ts_idx in tsubs:
            tsub_to_blocks[ts_idx].append(bi)
    # Convert to numpy arrays for faster operations
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
        
    # --- Initial Solution: Greedy Construction ---
    def greedy_construct(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubsets, dtype=bool)
        blocks_used = []
        
        # Gain for each block: number of uncovered t-subsets it covers
        # We can compute this on the fly or precompute updates.
        # For simplicity and speed in initial phase, we compute gain for all blocks initially,
        # then update incrementally.
        
        # Initial gain calculation
        gain = np.zeros(n_blocks, dtype=np.int32)
        # For each t-subset, if uncovered, it contributes 1 to the gain of every block covering it.
        # This is expensive to do from scratch for every step.
        # Instead, we pick a block, add it, and update gains of all other blocks.
        
        # Efficient initial gain:
        # gain[b] = sum(1 for ts in block_tsubs[b] if uncovered[ts])
        # Since all are uncovered initially, gain[b] = len(block_tsubs[b]) = C(k, t) for all b.
        # Wait, all blocks have the same size k, so they cover the same number of t-subsets.
        # So initial gain is constant for all blocks!
        # This means the first block choice is arbitrary (or random).
        
        # However, after picking one block, the gains change.
        
        current_blocks = []
        
        while np.any(uncovered):
            if time.time() > deadline - 100: # Reserve time for SA
                break
                
            # Find block with max gain
            # If gain is constant, pick random.
            # Let's just pick the one with max gain, breaking ties randomly.
            max_gain = gain.max()
            if max_gain == 0:
                break
                
            candidates = np.where(gain == max_gain)[0]
            chosen_idx = candidates[rng.randint(0, len(candidates) - 1)]
            
            current_blocks.append(chosen_idx)
            
            # Update uncovered
            tsubs = block_tsubs[chosen_idx]
            newly_covered = tsubs[uncovered[tsubs]]
            
            if len(newly_covered) > 0:
                uncovered[newly_covered] = False
                
                # Update gains for all blocks
                # For each newly covered t-subset, decrement gain of all blocks that cover it.
                # We need to know which blocks cover a t-subset.
                # Let's precompute tsub_to_blocks for this.
                pass
                
            # To avoid O(n_blocks * C(k,t)) update per step, we can just recalculate gains
            # if n_blocks is small enough, or use a smarter update.
            # n_blocks = C(19,7) = 50388.
            # C(7,4) = 35.
            # Recalculating gains for 50k blocks every step is 50k * 35 = 1.7M ops.
            # If we have ~150 steps, that's 250M ops. Might be too slow in Python.
            
            # Better approach: Maintain a list of blocks for each t-subset.
            # Let's precompute tsub_to_blocks once.
            
        return current_blocks


    # Re-implement greedy with proper gain updates
    def greedy_construct_fast(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubsets, dtype=bool)
        blocks_used = []
        
        # gain[bi] = number of uncovered t-subsets in block bi
        gain = np.zeros(n_blocks, dtype=np.int32)
        
        # Initialize gain
        # Since all tsubsets are uncovered, gain[bi] = len(block_tsubs[bi])
        # All blocks have same size, so gain is constant.
        gain.fill(len(block_tsubs[0]))
        
        while np.any(uncovered):
            if time.time() > deadline - 100:
                break
                
            max_gain = gain.max()
            if max_gain == 0:
                break
                
            candidates = np.where(gain == max_gain)[0]
            chosen_idx = candidates[rng.randint(0, len(candidates) - 1)]
            
            blocks_used.append(chosen_idx)
            
            # Update uncovered and gains
            tsubs = block_tsubs[chosen_idx]
            newly_covered = tsubs[uncovered[tsubs]]
            
            if len(newly_covered) > 0:
                uncovered[newly_covered] = False
                
                # Decrement gain for all blocks that cover any of the newly_covered t-subsets
                # We need to be careful not to decrement a block multiple times for the same t-subset
                # But a block either covers a t-subset or it doesn't.
                # If a block covers multiple newly_covered t-subsets, its gain should decrease by that count.
                
                # Efficient way:
                # For each newly covered t-subset ts, find all blocks covering ts.
                # For each such block b, decrement gain[b] by 1.
                
                # This can be slow if we do it in a loop.
                # Let's collect all affected block indices and use np.add.at
                
                affected_blocks = []
                for ts_idx in newly_covered:
                    affected_blocks.extend(tsub_to_blocks[ts_idx])
                
                if affected_blocks:
                    affected_blocks = np.array(affected_blocks, dtype=np.int32)
                    # We need to count how many times each block appears in affected_blocks
                    # because a single block might cover multiple newly covered t-subsets.
                    # np.add.at does exactly this: adds 1 for each occurrence.
                    np.add.at(gain, affected_blocks, -1)
            
            # Mark chosen block as unusable (gain = -1) to prevent re-selection
            gain[chosen_idx] = -1
            
        return blocks_used

    # Get initial solution
    initial_blocks = greedy_construct_fast(123)
    if not initial_blocks:
        # Fallback if greedy fails (shouldn't happen)
        return [list(b) for b in all_blocks[:100]] # Dummy

    # --- Simulated Annealing ---
    
    # State:
    # blocks: list of block indices
    # cover_count: numpy array of size n_tsubsets, count of how many times each t-subset is covered
    # block_tsubs: precomputed
    
    blocks = list(initial_blocks)
    cover_count = np.zeros(n_tsubsets, dtype=np.int32)
    
    # Initialize cover_count
    for bi in blocks:
        for ts_idx in block_tsubs[bi]:
            cover_count[ts_idx] += 1
            
    # Check validity (should be valid)
    # assert np.all(cover_count > 0)
    
    current_cost = len(blocks)
    best_cost = current_cost
    best_blocks = blocks.copy()
    
    # SA Parameters
    T = 50.0  # Initial temperature - lower to focus on exploitation
    T_min = 0.05
    alpha = 0.9995  # Slower cooling for more iterations at moderate T
    max_iterations = 500000  # Max iterations
    
    rng = random.Random(42)
    
    def get_block_idx(block_tuple):
        # We are working with indices, so this is just a passthrough if we pass index
        # But sometimes we might generate a block tuple.
        # For efficiency, we stick to indices.
        pass
        
    # Precompute a set of active blocks for O(1) lookup? 
    # active_set = set(blocks)
    
    # Helper to get unique t-subsets for a block
    # unique_tsubs_for_block[bi] = [ts for ts in block_tsubs[bi] if cover_count[ts] == 1]
    # This is expensive to compute for all blocks. Compute on demand.
    
    iteration = 0
    while T > T_min and time.time() < deadline - 5:
        iteration += 1
        
        # Choose a move type
        # 1. Remove a redundant block
        # 2. Swap a block
        # 3. Add a random block (to escape local minima)
        
        move_type = rng.random()
        
        if move_type < 0.3:
            # Try to remove a redundant block
            # Find a block that is redundant (all its t-subsets have cover_count > 1)
            # We can pick a random block and check if it's redundant.
            if not blocks:
                continue
                
            i = rng.randint(0, len(blocks) - 1)
            bi = blocks[i]
            
            # Check if redundant using vectorized numpy
            block_ts = block_tsubs[bi]
            redundant = np.all(cover_count[block_ts] > 1)
            
            if redundant:
                # Remove it
                blocks.pop(i)
                for ts_idx in block_tsubs[bi]:
                    cover_count[ts_idx] -= 1
                current_cost -= 1
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = blocks.copy()
                    

        elif move_type < 0.8:
            # Try a swap
            if not blocks:
                continue
                
            i = rng.randint(0, len(blocks) - 1)
            bi_old = blocks[i]
            
            # Find a new block bi_new
            # Ideally, bi_new should cover the unique t-subsets of bi_old
            # Unique t-subsets of bi_old:
            unique_tsubs = []
            for ts_idx in block_tsubs[bi_old]:
                if cover_count[ts_idx] == 1:
                    unique_tsubs.append(ts_idx)
                    
            if not unique_tsubs:
                # bi_old is redundant, remove it instead of swapping
                blocks.pop(i)
                for ts_idx in block_tsubs[bi_old]:
                    cover_count[ts_idx] -= 1
                current_cost -= 1
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = blocks.copy()
                    
            else:
                # We need a block that covers all unique_tsubs.
                # Strategy: Find candidate blocks that cover the most unique t-subsets.
                # For efficiency, pick the first unique t-subset and look at blocks covering it.
                # Then filter those that cover all unique_tsubs.
                
                # Get blocks that cover the first unique t-subset
                first_ts = unique_tsubs[0]
                candidate_blocks = tsub_to_blocks_np[first_ts]
                
                # Filter candidates: must cover ALL unique_tsubs and not be bi_old
                valid_candidates = []
                for cand in candidate_blocks:
                    if cand == bi_old:
                        continue
                    # Check if cand covers all unique_tsubs
                    cand_ts = block_tsubs[cand]
                    # Check coverage: for each ts in unique_tsubs, is it in cand_ts?
                    # Use numpy: create a set for faster lookup
                    cand_ts_set = set(cand_ts.tolist())
                    if all(ts in cand_ts_set for ts in unique_tsubs):
                        valid_candidates.append(cand)
                
                if not valid_candidates:
                    continue
                    
                # Pick a random valid candidate
                bi_new = valid_candidates[rng.randint(0, len(valid_candidates) - 1)]
                
                # Perform swap
                # Remove bi_old
                for ts_idx in block_tsubs[bi_old]:
                    cover_count[ts_idx] -= 1
                # Add bi_new
                for ts_idx in block_tsubs[bi_new]:
                    cover_count[ts_idx] += 1
                    
                blocks[i] = bi_new
                # Cost doesn't change
                
        else:
            # Add a random block
            # This increases cost, but might enable future removals.
            # We should only do this if we are stuck (high T).
            if T < 10.0:
                continue # Don't add blocks when cold
                
            bi_new = rng.randint(0, n_blocks - 1)
            
            # Avoid adding duplicate blocks? 
            # Duplicates don't help coverage but waste space.
            # If bi_new is already in blocks, skip.
            if bi_new in blocks:
                continue
                
            blocks.append(bi_new)
            for ts_idx in block_tsubs[bi_new]:
                cover_count[ts_idx] += 1
            current_cost += 1
            
        # Update temperature
        T *= alpha
        
        # If we improved, we can reset T slightly or just keep cooling
        # Standard SA keeps cooling.
        
    # Final Polish: Multiple passes of greedy removal with different orderings
    def polish_solution(block_list, max_time=10.0):
        """Remove redundant blocks using multiple passes with different orderings."""
        if not block_list:
            return block_list
        
        cover_count_polish = np.zeros(n_tsubsets, dtype=np.int32)
        for bi in block_list:
            cover_count_polish[block_tsubs[bi]] += 1
        
        current = list(block_list)
        polish_start = time.time()
        
        # Multiple passes with different random orderings
        for pass_num in range(5):
            if time.time() - polish_start > max_time:
                break
                
            # Shuffle the order for this pass
            rng_polish = random.Random(12345 + pass_num)
            indices = list(range(len(current)))
            rng_polish.shuffle(indices)
            
            i = 0
            while i < len(current):
                if time.time() - polish_start > max_time:
                    break
                    
                idx = indices[i] if i < len(indices) else i
                if idx >= len(current):
                    break
                    
                bi = current[idx]
                # Check redundancy using vectorized numpy
                block_ts = block_tsubs[bi]
                if np.all(cover_count_polish[block_ts] > 1):
                    # Remove it
                    for ts_idx in block_ts:
                        cover_count_polish[ts_idx] -= 1
                    current.pop(idx)
                    # Adjust indices
                    for j in range(i, len(indices)):
                        if indices[j] > idx:
                            indices[j] -= 1
                    indices.pop(i)
                    # Don't increment i
                else:
                    i += 1
        
        return current
    
    # Apply polish to best solution
    polished = polish_solution(best_blocks, max_time=15.0)
    
    if len(polished) < best_cost:
        best_blocks = polished
        best_cost = len(polished)
        
    return [list(all_blocks[bi]) for bi in best_blocks]
