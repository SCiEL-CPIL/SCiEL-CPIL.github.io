import itertools
import time
import random
import numpy as np

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
    # This is useful for checking redundancy quickly if we had a set of active blocks,
    # but since we track coverage counts, we mainly need block_tsubs.
    # However, for "which blocks cover this t-subset" we might need it for advanced moves.
    # For now, we rely on coverage counts.
    
    def get_block_tsubs(block):
        """Returns the array of t-subset indices for a given block tuple."""
        # We can't use a dict for tuples of ints easily if we want speed, 
        # but we have all_blocks list. 
        # Actually, in SA we deal with block indices (0 to n_blocks-1).
        # So we just return block_tsubs[block_idx]
        pass
        
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

    # Precompute tsub_to_blocks: list of block indices for each t-subset
    # This is a list of lists, or a list of numpy arrays.
    tsub_to_blocks = [[] for _ in range(n_tsubsets)]
    for bi, tsubs in enumerate(block_tsubs):
        for ts_idx in tsubs:
            tsub_to_blocks[ts_idx].append(bi)
    # Convert to numpy arrays for faster iteration? Or keep as lists.
    # Lists are fine for iteration.
    
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
    T = 100.0  # Initial temperature
    T_min = 0.1
    alpha = 0.999  # Cooling rate
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
            
            # Check if redundant
            redundant = True
            for ts_idx in block_tsubs[bi]:
                if cover_count[ts_idx] == 1:
                    redundant = False
                    break
                    
            if redundant:
                # Remove it
                blocks.pop(i)
                for ts_idx in block_tsubs[bi]:
                    cover_count[ts_idx] -= 1
                current_cost -= 1
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = blocks.copy()
                    
            else:
                # Not redundant, do nothing (or try swap)
                pass
                
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
                # bi_old is redundant, we should have removed it in the other branch.
                # But maybe we missed it. Let's just try to swap it out for something else
                # to potentially improve structure, or just remove it.
                # Actually, if it's redundant, removal is better.
                # Let's force a removal attempt if we end up here with a redundant block.
                # But for now, let's just pick a random new block.
                bi_new = rng.randint(0, n_blocks - 1)
            else:
                # We need a block that covers all unique_tsubs.
                # This is a constraint.
                # We can sample some blocks and check.
                # Or, we can just pick a random block and accept if it doesn't break coverage.
                # If we swap bi_old with bi_new, we must ensure no t-subset becomes uncovered.
                # A t-subset becomes uncovered if cover_count[ts] was 1 and bi_old covered it, 
                # and bi_new does NOT cover it.
                
                # Strategy: Pick a random bi_new.
                # Check if it's a valid swap.
                # Valid if: for all ts in bi_old, if cover_count[ts] == 1, then ts in bi_new.
                
                bi_new = rng.randint(0, n_blocks - 1)
                
                if bi_new == bi_old:
                    continue
                    
                # Check validity
                valid = True
                for ts_idx in block_tsubs[bi_old]:
                    if cover_count[ts_idx] == 1:
                        # ts_idx is uniquely covered by bi_old.
                        # bi_new must cover it.
                        if ts_idx not in block_tsubs[bi_new]:
                            # We need a fast check. 
                            # block_tsubs[bi_new] is a numpy array. 'in' is O(N).
                            # N=35. This is fine.
                            valid = False
                            break
                            
                if not valid:
                    continue
                    
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
        
    # Final Polish: Remove any remaining redundant blocks
    changed = True
    while changed:
        changed = False
        for i in range(len(best_blocks)):
            bi = best_blocks[i]
            # Rebuild cover_count for best_blocks to check redundancy?
            # This is expensive. 
            # Let's just do a few passes of removal on the current `blocks` state 
            # if it's better, or on `best_blocks`.
            pass
            
    # Let's do a final greedy reduction on the best found solution
    final_blocks = best_blocks.copy()
    # Rebuild cover count for final_blocks
    cover_count_final = np.zeros(n_tsubsets, dtype=np.int32)
    for bi in final_blocks:
        for ts_idx in block_tsubs[bi]:
            cover_count_final[ts_idx] += 1
            
    # Remove redundant
    i = 0
    while i < len(final_blocks):
        bi = final_blocks[i]
        redundant = True
        for ts_idx in block_tsubs[bi]:
            if cover_count_final[ts_idx] == 1:
                redundant = False
                break
        if redundant:
            final_blocks.pop(i)
            for ts_idx in block_tsubs[bi]:
                cover_count_final[ts_idx] -= 1
            # Don't increment i
        else:
            i += 1
            
    if len(final_blocks) < best_cost:
        best_blocks = final_blocks
        best_cost = len(final_blocks)
        
    return [list(all_blocks[bi]) for bi in best_blocks]
