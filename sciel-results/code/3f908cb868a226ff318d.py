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
    # Parameters specific to this task: v=19, k=8, t=5
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index for fast lookup
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute the list of t-subset indices for every possible k-subset (block)
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
            
    # Deadline: 150s limit, use 145s for safety
    start_time = time.time()
    deadline = start_time + 145.0
    
    def get_elapsed():
        return time.time() - start_time

    # GRASP: Greedy Randomized Adaptive Search Procedure
    def grasp_construct(alpha=0.3):
        """
        Construct a solution using GRASP.
        At each step, compute the gain (new t-subsets covered) for each candidate block.
        Select from the restricted candidate list (top fraction alpha of best gains) randomly.
        """
        uncovered = bytearray([1] * num_tsubs)
        uncovered_count = num_tsubs
        selected_blocks = []
        
        while uncovered_count > 0:
            if get_elapsed() > deadline - 10.0:
                return selected_blocks
            
            # Pick a random uncovered t-subset to anchor candidate selection
            if uncovered_count < 500:
                uncov_indices = [i for i in range(num_tsubs) if uncovered[i]]
                if not uncov_indices:
                    break
                s_idx = random.choice(uncov_indices)
            else:
                while True:
                    s_idx = random.randint(0, num_tsubs - 1)
                    if uncovered[s_idx]:
                        break
            
            # Get all blocks that cover this t-subset
            candidates = tsub_blocks[s_idx]
            if not candidates:
                break
            
            # Compute gain for each candidate block
            gains = []
            for b_idx in candidates:
                gain = 0
                for j in block_tsubs[b_idx]:
                    if uncovered[j]:
                        gain += 1
                        if gain == 56:
                            break
                gains.append((gain, b_idx))
            
            if not gains:
                break
            
            # Sort by gain descending
            gains.sort(key=lambda x: -x[0])
            max_gain = gains[0][0]
            
            # Restricted candidate list: top alpha fraction with gain close to max
            threshold = max_gain * (1.0 - alpha)
            rcl = [(g, b) for g, b in gains if g >= threshold]
            
            # Randomly select from RCL
            selected_gain, selected_block = random.choice(rcl)
            
            if selected_block == -1:
                break
                
            selected_blocks.append(selected_block)
            
            for j in block_tsubs[selected_block]:
                if uncovered[j]:
                    uncovered[j] = 0
                    uncovered_count -= 1
                    
        return selected_blocks

    # Local Search: 1-swap refinement
    def local_search(block_indices):
        """
        Refine a solution via 1-swap local search.
        Try removing each block and see if coverage is still maintained.
        If not, try swapping with a block that covers more uncovered t-subsets.
        """
        block_indices = list(set(block_indices))
        
        changed = True
        while changed:
            changed = False
            if get_elapsed() > deadline - 5.0:
                break
                
            # Compute cover counts
            cover_count = [0] * num_tsubs
            for b_idx in block_indices:
                for j in block_tsubs[b_idx]:
                    cover_count[j] += 1
            
            new_block_indices = []
            for b_idx in block_indices:
                # Check if this block is essential (covers any t-subset with count 1)
                is_essential = False
                for j in block_tsubs[b_idx]:
                    if cover_count[j] == 1:
                        is_essential = True
                        break
                
                if is_essential:
                    new_block_indices.append(b_idx)
                else:
                    # Try to find a replacement block that covers more uncovered t-subsets
                    # First, mark this block's t-subsets as potentially uncovered
                    temp_cover = cover_count.copy()
                    for j in block_tsubs[b_idx]:
                        temp_cover[j] -= 1
                    
                    # Find t-subsets that would become uncovered
                    newly_uncovered = [j for j in block_tsubs[b_idx] if temp_cover[j] == 0]
                    
                    if not newly_uncovered:
                        # Safe to remove
                        changed = True
                        continue
                    
                    # Find a block that covers the most newly uncovered t-subsets
                    best_replace = -1
                    best_new_gain = -1
                    for j in newly_uncovered:
                        for cand in tsub_blocks[j]:
                            if cand in block_indices:
                                continue
                            # Count how many newly_uncovered t-subsets this block covers
                            new_gain = 0
                            for nj in newly_uncovered:
                                if nj in block_tsubs[cand]:
                                    new_gain += 1
                            if new_gain > best_new_gain:
                                best_new_gain = new_gain
                                best_replace = cand
                            if new_gain == len(newly_uncovered):
                                break
                        if best_new_gain == len(newly_uncovered):
                            break
                    
                    if best_replace != -1:
                        new_block_indices.append(best_replace)
                        changed = True
                    else:
                        new_block_indices.append(b_idx)
            
            block_indices = new_block_indices
            
        return block_indices

    # Run initial GRASP
    initial_blocks = grasp_construct(alpha=0.3)
    optimized_blocks = local_search(initial_blocks)
    
    best_len = len(optimized_blocks)
    best_block_list = [all_blocks[i] for i in optimized_blocks]
    
    # Random Restarts with varying alpha
    alphas = [0.2, 0.3, 0.4, 0.5, 0.6]
    for restart in range(100):
        if get_elapsed() > deadline - 10.0:
            break
            
        alpha = random.choice(alphas)
        current_blocks = grasp_construct(alpha=alpha)
        if not current_blocks:
            continue
            
        optimized = local_search(current_blocks)
        
        if len(optimized) < best_len:
            best_len = len(optimized)
            best_block_list = [all_blocks[i] for i in optimized]
            
            if best_len <= 325:
                break

    return [list(b) for b in best_block_list]
