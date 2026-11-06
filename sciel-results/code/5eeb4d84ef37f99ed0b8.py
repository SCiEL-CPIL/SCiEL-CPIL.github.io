import itertools
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Generate all t-subsets and map them to indices
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    num_t_subs = len(all_t_subs)
    
    # 2. Generate all k-subsets (blocks) and pre-compute their coverage
    all_k_subs = list(itertools.combinations(range(v), k))
    
    # Pre-compute coverage: for each block, list of indices of t-subsets it covers
    block_coverages = []
    for block in all_k_subs:
        # Get all t-subsets within this block
        cov_indices = []
        for t_sub in itertools.combinations(block, t):
            cov_indices.append(t_sub_to_idx[t_sub])
        block_coverages.append(cov_indices)
        
    # 3. Greedy Set Cover
    # counts[i] is the number of blocks covering t-subset i
    counts = [0] * num_t_subs
    selected_blocks = []
    selected_indices = []
    
    while any(c == 0 for c in counts):
        if time.time() > deadline:
            break
            
        best_block_idx = -1
        best_gain = -1
        
        # Iterate through all possible blocks to find the one with max gain
        for i, cov in enumerate(block_coverages):
            # Calculate gain: number of t-subsets in cov that are currently uncovered
            gain = 0
            for idx in cov:
                if counts[idx] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
                if gain == len(cov): # Perfect gain, can't do better
                    break
                    
        if best_gain <= 0:
            # Should not happen if covering is possible, but safety check
            break
            
        # Add the best block
        selected_indices.append(best_block_idx)
        selected_blocks.append(all_k_subs[best_block_idx])
        
        # Update counts
        for idx in block_coverages[best_block_idx]:
            counts[idx] += 1
            
    # 4. Redundancy Removal (Post-processing)
    # We try to remove blocks that are not essential
    # We need to be careful with the counts array as we modify it
    # Strategy: For each selected block in reverse order (or any order), 
    # tentatively remove it, check if coverage is broken, if not, keep it removed.
    
    # To make this efficient, we can just re-verify coverage after removal.
    # Since the number of blocks is small (~150), we can afford O(N * K * T) checks.
    
    final_blocks = []
    # We need a robust way to check if a set of blocks covers all t-subsets.
    # We can maintain a global count array and decrement/increment as we remove/add.
    
    # Let's restart counts based on current selected_blocks
    counts = [0] * num_t_subs
    for b_idx in selected_indices:
        for idx in block_coverages[b_idx]:
            counts[idx] += 1
            
    # Try to remove each block
    # We iterate multiple times until no more removals are possible
    # Or just one pass in reverse order is often sufficient for greedy results
    for i in range(len(selected_indices) - 1, -1, -1):
        b_idx = selected_indices[i]
        cov = block_coverages[b_idx]
        
        # Check if any t-subset becomes uncovered
        is_essential = False
        for idx in cov:
            if counts[idx] == 1:
                is_essential = True
                break
                
        if is_essential:
            # Keep the block
            final_blocks.append(all_k_subs[b_idx])
        else:
            # Remove the block: decrement counts
            for idx in cov:
                counts[idx] -= 1
            # Do not add to final_blocks
            
    # Reverse to maintain original order (optional, but nice)
    final_blocks.reverse()
    
    return final_blocks

# Note: The above redundancy removal is a single pass. 
# A more aggressive multi-pass removal could be added if time permits, 
# but single pass usually captures most redundancies in greedy constructions.
# Given the strict time limit, single pass is safer.
