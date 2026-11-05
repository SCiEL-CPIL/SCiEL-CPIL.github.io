import itertools
import time
import random
import sys
import math

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each is a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def check_time(budget=5.0):
        return time.time() - start_time > deadline - budget

    # 1. Precomputation
    # Generate all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    # Generate all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # block_tsubs[i]: list of indices of t-subsets contained in block i
    block_tsubs = [[] for _ in range(num_blocks)]
    # tsub_blocks[j]: list of indices of blocks containing t-subset j
    tsub_blocks = [[] for _ in range(num_tsubs)]

    for i, b in enumerate(all_blocks):
        for ts in itertools.combinations(b, t):
            j = tsub_index[ts]
            block_tsubs[i].append(j)
            tsub_blocks[j].append(i)

    # 2. Greedy Initialization with Dynamic Counting
    def greedy_cover():
        counts = [0] * num_tsubs
        selected_indices = set()
        uncovered = set(range(num_tsubs))
        
        # Pre-sort blocks by size of block_tsubs? They are all same size C(k,t).
        # We need to find the block that covers the most uncovered subsets.
        
        while uncovered:
            if check_time(10.0):
                break
                
            best_block = -1
            best_gain = -1
            
            # Optimization: If the number of remaining blocks is large, 
            # checking all is O(N_blocks * C(k,t)). 
            # For v=19, N_blocks=75k, C(8,5)=56. 75k * 56 = 4.2M ops per step.
            # Number of steps ~ 300. Total 1.2B ops. Too slow in Python.
            
            # We need a faster greedy.
            # Approximation: Pick a random uncovered t-subset, 
            # then pick a random block covering it? No, that's weak.
            
            # Better: Maintain a heap or list of "potential" blocks?
            # Or: Randomized Greedy.
            
            # Let's try a different greedy: 
            # 1. Pick a random uncovered t-subset 'u'.
            # 2. From all blocks covering 'u', pick the one that covers the most OTHER uncovered subsets.
            
            if not uncovered:
                break
                
            u = random.choice(list(uncovered))
            candidates = tsub_blocks[u]
            
            best_gain = -1
            best_block = -1
            
            for b_idx in candidates:
                if b_idx in selected_indices:
                    continue
                gain = 0
                for ts_idx in block_tsubs[b_idx]:
                    if ts_idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block == -1:
                # Should not happen if valid parameters
                break
                
            selected_indices.add(best_block)
            for ts_idx in block_tsubs[best_block]:
                if counts[ts_idx] == 0:
                    uncovered.discard(ts_idx)
                counts[ts_idx] += 1
                
        return list(selected_indices), counts

    # 3. Local Search: 1-Remove and 1-Swap
    def local_search(indices, counts):
        indices = set(indices)
        counts = list(counts) # Copy
        improved = True
        iter_count = 0
        
        while improved and not check_time(5.0):
            improved = False
            iter_count += 1
            if iter_count > 100: # Safety loop limit
                break
                
            # Pass 1: Removal
            # Check if any block can be removed
            for b_idx in list(indices):
                # Check if any t-subset in this block has count == 1
                can_remove = True
                for ts_idx in block_tsubs[b_idx]:
                    if counts[ts_idx] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    indices.discard(b_idx)
                    for ts_idx in block_tsubs[b_idx]:
                        counts[ts_idx] -= 1
                    improved = True
                    # Restart loop to re-evaluate other blocks
                    break 
            
            if improved:
                continue
                
            # Pass 2: Swap
            # For each block, find 'lost' subsets (count==1)
            # Find a replacement block that covers all 'lost' subsets
            # Only try if len(lost) is small to keep intersection fast
            
            for b_idx in list(indices):
                lost = []
                for ts_idx in block_tsubs[b_idx]:
                    if counts[ts_idx] == 1:
                        lost.append(ts_idx)
                
                if not lost:
                    continue # Handled by removal
                
                # If too many lost, skip swap (hard to find replacement)
                if len(lost) > 3:
                    continue
                    
                # Find intersection of blocks covering all lost subsets
                # Start with the smallest candidate list
                min_list_len = float('inf')
                min_list = None
                for ts_idx in lost:
                    if len(tsub_blocks[ts_idx]) < min_list_len:
                        min_list_len = len(tsub_blocks[ts_idx])
                        min_list = tsub_blocks[ts_idx]
                        
                if not min_list:
                    continue

                # Filter min_list to only contain blocks that cover ALL other lost subsets
                candidates = []
                for cand_idx in min_list:
                    if cand_idx in indices:
                        continue
                    # Check if cand_idx covers all other lost
                    valid = True
                    cand_tsubs_set = set(block_tsubs[cand_idx])
                    for other_ts in lost:
                        if other_ts not in cand_tsubs_set:
                            valid = False
                            break
                    if valid:
                        candidates.append(cand_idx)
                        
                if candidates:
                    # Pick a random candidate (or the one with best "extra" coverage?)
                    # For now, random
                    new_block = random.choice(candidates)
                    
                    # Perform Swap
                    indices.discard(b_idx)
                    indices.add(new_block)
                    
                    # Update counts
                    for ts_idx in block_tsubs[b_idx]:
                        counts[ts_idx] -= 1
                    for ts_idx in block_tsubs[new_block]:
                        counts[ts_idx] += 1
                        
                    improved = True
                    break # Restart loop

        return list(indices), counts

    # Main Execution
    # Run multiple restarts of Greedy + Local Search
    best_indices = []
    best_len = float('inf')
    
    num_restarts = 10
    
    for r in range(num_restarts):
        if check_time(15.0):
            break
            
        # Greedy
        indices, counts = greedy_cover()
        
        if not indices:
            continue
            
        # Local Search
        indices, counts = local_search(indices, counts)
        
        if len(indices) < best_len:
            best_len = len(indices)
            best_indices = indices
            
        # Optional: Perturbation and Re-optimize
        # If we have time, perturb best solution
        if not check_time(30.0) and best_indices:
            # Remove 2 random blocks and repair?
            # Simpler: Just run local search again on a copy?
            # Or, swap one random block with another random block and run local search?
            pass

    return [list(all_blocks[i]) for i in best_indices]

# To ensure the function is available at module level as required
# The task says "Keep the function names and signatures of the seed program."
# The seed program defines construct(v, k, t).
# I have defined construct above.
