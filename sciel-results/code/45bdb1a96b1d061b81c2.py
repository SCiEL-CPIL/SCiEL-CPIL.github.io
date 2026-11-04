import itertools
import time
import random
import sys
import math
import numpy as np

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each is a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    def get_elapsed():
        return time.time() - start_time

    # Precomputation
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # block_tsubs[i] is a list of indices into all_tsubs
    block_tsubs = []
    for b in all_blocks:
        tsubs_in_b = [tsub_index[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(tsubs_in_b)
        
    # Inverse mapping: for each t-subset, which blocks cover it?
    tsub_blocks = [[] for _ in range(num_tsubs)]
    for i, b_tsubs in enumerate(block_tsubs):
        for j in b_tsubs:
            tsub_blocks[j].append(i)
            
    # Fast membership check: use bitsets for speed? 
    # For now, keep lists but optimize the counting.

    def greedy_construct(seed=None):
        if seed is not None:
            random.seed(seed)
        uncovered = bytearray([1] * num_tsubs)
        uncovered_count = num_tsubs
        selected_blocks = []
        
        while uncovered_count > 0:
            if get_elapsed() > deadline - 5.0:
                break
            
            # Pick a random uncovered t-subset
            if uncovered_count < 100:
                uncov_indices = [i for i in range(num_tsubs) if uncovered[i]]
                if not uncov_indices:
                    break
                s_idx = random.choice(uncov_indices)
            else:
                while True:
                    s_idx = random.randint(0, num_tsubs - 1)
                    if uncovered[s_idx]:
                        break
            
            candidates = tsub_blocks[s_idx]
            best_block_idx = -1
            best_gain = -1
            
            for b_idx in candidates:
                gain = 0
                for j in block_tsubs[b_idx]:
                    if uncovered[j]:
                        gain += 1
                        if gain == 56:
                            break
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx == -1:
                break
                
            selected_blocks.append(best_block_idx)
            
            for j in block_tsubs[best_block_idx]:
                if uncovered[j]:
                    uncovered[j] = 0
                    uncovered_count -= 1
                    
        return selected_blocks

    def reduce_blocks(block_indices, check_time=True):
        block_indices = list(set(block_indices))
        if not block_indices:
            return []
        
        # Use numpy for coverage counts
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b_idx in block_indices:
            cover_count[block_tsubs[b_idx]] += 1
        
        changed = True
        while changed:
            changed = False
            if check_time and get_elapsed() > deadline - 2.0:
                break
            
            new_block_indices = []
            for b_idx in block_indices:
                # Check if any t-subset is uniquely covered by b_idx
                tsubs_in_b = block_tsubs[b_idx]
                # Vectorized check: are any of these tsubs having count == 1?
                # Note: cover_count is updated incrementally? 
                # For efficiency in the inner loop, let's just check the current state.
                # If we remove a block, counts change.
                
                # Optimization: 
                # If min(cover_count[tsubs_in_b]) > 1, then b_idx is redundant.
                if np.all(cover_count[tsubs_in_b] > 1):
                    # Redundant, remove it
                    cover_count[tsubs_in_b] -= 1
                    changed = True
                else:
                    new_block_indices.append(b_idx)
            
            block_indices = new_block_indices
            
        return block_indices

    def swap_optimization(block_indices):
        """
        Try to replace a block with another that covers the same unique t-subsets 
        but allows for further reduction or has better properties.
        Here, we focus on swapping blocks that are "critical" (have unique tsubs) 
        with blocks that share more of those tsubs with other blocks, potentially 
        making other blocks redundant.
        """
        if not block_indices:
            return block_indices
        
        block_indices = list(block_indices)
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b_idx in block_indices:
            cover_count[block_tsubs[b_idx]] += 1
            
        improved = True
        while improved:
            improved = False
            if get_elapsed() > deadline - 2.0:
                break
                
            for i, b_idx in enumerate(block_indices):
                tsubs_in_b = block_tsubs[b_idx]
                unique_tsubs = np.where(cover_count[tsubs_in_b] == 1)[0]
                
                if len(unique_tsubs) == 0:
                    continue # Should have been removed by reduce, but just in case
                    
                # We want to find a replacement block that covers all unique_tsubs
                # and potentially covers more "shared" tsubs to help others.
                
                # Find blocks that cover ALL unique_tsubs
                # This is an intersection problem.
                # Since unique_tsubs can be large, this might be slow.
                # Heuristic: Pick one unique tsub, find blocks covering it, filter.
                
                if len(unique_tsubs) > 3:
                    continue # Too hard to satisfy all, skip for now
                    
                target_tsub_idx = tsubs_in_b[unique_tsubs[0]]
                candidates = tsub_blocks[target_tsub_idx]
                
                best_swap_gain = 0
                best_swap_block = -1
                
                for c_idx in candidates:
                    if c_idx == b_idx or c_idx in block_indices:
                        continue
                        
                    # Check if c_idx covers all unique_tsubs
                    covers_all = True
                    for ut_idx in unique_tsubs:
                        if ut_idx not in block_tsubs[c_idx]:
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    
                    # Calculate gain: How many new tsubs does c_idx cover that were 
                    # previously covered by exactly 1 block (b_idx), or how many 
                    # does it cover that are shared?
                    
                    # Actually, the goal is to enable removal of other blocks.
                    # If we swap b_idx for c_idx:
                    # 1. Subtract b_idx from cover_count
                    # 2. Add c_idx to cover_count
                    # 3. See if any OTHER block becomes redundant.
                    
                    # This simulation is expensive. 
                    # Simpler heuristic: Prefer c_idx that covers more total tsubs 
                    # that are currently "tight" (count <= 2).
                    
                    gain = 0
                    for j in block_tsubs[c_idx]:
                        if cover_count[j] <= 2:
                            gain += 1
                            
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap_block = c_idx
                        
                if best_swap_block != -1:
                    # Perform swap
                    cover_count[block_tsubs[b_idx]] -= 1
                    block_indices[i] = best_swap_block
                    cover_count[block_tsubs[best_swap_block]] += 1
                    improved = True
                    break # Restart loop as indices changed
                
        return block_indices

    # Initial Greedy
    initial_blocks = greedy_construct()
    initial_blocks = reduce_blocks(initial_blocks)
    
    best_len = len(initial_blocks)
    best_block_list = [all_blocks[i] for i in initial_blocks]
    
    # Try swap optimization
    if get_elapsed() < deadline - 10.0:
        swapped_blocks = swap_optimization(initial_blocks)
        swapped_blocks = reduce_blocks(swapped_blocks)
        if len(swapped_blocks) < best_len:
            best_len = len(swapped_blocks)
            best_block_list = [all_blocks[i] for i in swapped_blocks]

    # Multiple restarts
    num_restarts = 10
    restarts_done = 0
    while restarts_done < num_restarts and get_elapsed() < deadline - 20.0:
        random.seed()
        new_blocks = greedy_construct()
        new_blocks = reduce_blocks(new_blocks)
        
        if get_elapsed() < deadline - 10.0:
            new_blocks = swap_optimization(new_blocks)
            new_blocks = reduce_blocks(new_blocks)
            
        if len(new_blocks) < best_len:
            best_len = len(new_blocks)
            best_block_list = [all_blocks[i] for i in new_blocks]
            
        restarts_done += 1

    return [list(b) for b in best_block_list]
