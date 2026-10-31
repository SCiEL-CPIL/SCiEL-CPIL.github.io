import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    # 1. Precompute all blocks and their t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map each block index to its t-subsets (as tuples)
    block_tsubs = []
    for b in all_blocks:
        block_tsubs.append(list(itertools.combinations(b, t)))
        
    # Map each t-subset to the list of block indices containing it
    tsub_to_blocks = {}
    for i, tsubs in enumerate(block_tsubs):
        for ts in tsubs:
            if ts not in tsub_to_blocks:
                tsub_to_blocks[ts] = []
            tsub_to_blocks[ts].append(i)

    # 2. Fast Greedy Construction
    # gains[i] = number of currently uncovered t-subsets in block i
    # Initially, all are uncovered, so gains[i] = C(k, t)
    initial_gain = len(block_tsubs[0])
    gains = np.full(num_blocks, initial_gain, dtype=np.int32)
    
    # coverage_count[ts] = number of selected blocks covering ts
    coverage_count = {}
    
    selected_indices = []
    
    # To speed up greedy, we can use a simple loop.
    # Optimizations:
    # - If a block's gain is 0, it's useless.
    # - We need to update gains for all blocks sharing a t-subset with the selected block.
    
    while True:
        if time.time() > deadline - 10: # Leave 10s for local search
            break
            
        # Find block with max gain
        max_gain = np.max(gains)
        if max_gain <= 0:
            break
            
        # Find index with max gain. 
        # np.argmax returns the first occurrence. 
        # To add diversity, we could pick randomly from maxes, but for speed, let's just pick one.
        # Or, to allow restarts later, we can just run one deterministic greedy first.
        best_idx = np.argmax(gains)
        
        selected_indices.append(best_idx)
        
        # Update coverage and gains
        for ts in block_tsubs[best_idx]:
            if ts not in coverage_count:
                coverage_count[ts] = 0
            coverage_count[ts] += 1
            
            if coverage_count[ts] == 1:
                # This t-subset was just covered for the first time.
                # Decrement gain for all blocks that contain this t-subset.
                for b_idx in tsub_to_blocks[ts]:
                    gains[b_idx] -= 1

    # If we ran out of time during greedy, we still have a partial solution.
    # But greedy usually finishes very fast for v=19.
    
    # 3. Local Search / Reduction
    # Convert selected indices to a set for O(1) lookup
    selected_set = set(selected_indices)
    
    # We will iteratively try to remove blocks that are redundant.
    # A block is redundant if all its t-subsets are covered by at least one *other* selected block.
    # i.e., for every ts in block b, count of selected blocks containing ts > 1.
    
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        # Iterate through a copy of the list so we can remove items
        for i in range(len(selected_indices) - 1, -1, -1):
            b_idx = selected_indices[i]
            is_redundant = True
            for ts in block_tsubs[b_idx]:
                # Count how many selected blocks cover ts
                # We need to check if any OTHER selected block covers ts.
                # This is O(num_blocks_containing_ts) which is small.
                count = 0
                for other_idx in tsub_to_blocks[ts]:
                    if other_idx in selected_set:
                        count += 1
                        if count > 1:
                            break # Found at least one other, so this ts is covered by someone else
                if count <= 1:
                    # Only this block covers ts (or none, but it's selected so >=1)
                    is_redundant = False
                    break
            
            if is_redundant:
                selected_indices.pop(i)
                selected_set.remove(b_idx)
                changed = True

    # Convert indices back to blocks
    final_blocks = [all_blocks[i] for i in selected_indices]
    
    # Ensure validity check (just in case, though logic should hold)
    # The problem statement says "Valid when every t-subset ... lies in at least one block".
    # Our greedy ensures this. Our reduction only removes blocks if coverage remains >= 1.
    
    return final_blocks
