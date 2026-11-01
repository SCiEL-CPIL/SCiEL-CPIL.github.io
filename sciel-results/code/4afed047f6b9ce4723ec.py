import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for 120s limit
    
    # 1. Precomputation
    # Generate all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Generate all k-subsets (blocks)
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # For each block, store the indices of the t-subsets it covers
    block_t_indices = []
    for block in all_k_subsets:
        indices = [t_subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_t_indices.append(indices)
        
    # For each t-subset, store the list of blocks that cover it
    t_subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_blocks[idx].append(b_idx)

    # 2. Greedy Set Cover
    uncovered = [True] * num_t_subsets
    gain = [0] * num_k_subsets
    
    # Initialize gain
    for b_idx in range(num_k_subsets):
        gain[b_idx] = len(block_t_indices[b_idx])
        
    selected_blocks = []
    
    while True:
        if time.time() > deadline:
            break
        max_gain = max(gain)
        if max_gain == 0:
            break
            
        best_block = gain.index(max_gain)
        
        selected_blocks.append(best_block)
        
        # Update gains
        for t_idx in block_t_indices[best_block]:
            if uncovered[t_idx]:
                uncovered[t_idx] = False
                for b_idx in t_subset_blocks[t_idx]:
                    gain[b_idx] -= 1
                    
    # 3. Minimalization (Local Search: Deletion)
    # Calculate cover counts
    cover_count = [0] * num_t_subsets
    for b_idx in selected_blocks:
        for t_idx in block_t_indices[b_idx]:
            cover_count[t_idx] += 1
            
    active_blocks = set(selected_blocks)
    
    # Multiple passes of deletion to find more removable blocks
    # We do up to 3 passes, or until no more blocks can be removed
    for _pass in range(3):
        if time.time() > deadline:
            break
        indices_to_check = list(active_blocks)
        random.shuffle(indices_to_check)
        removed_any = False
        
        for b_idx in indices_to_check:
            if b_idx not in active_blocks:
                continue
                
            can_remove = True
            for t_idx in block_t_indices[b_idx]:
                if cover_count[t_idx] <= 1:
                    can_remove = False
                    break
                    
            if can_remove:
                active_blocks.remove(b_idx)
                for t_idx in block_t_indices[b_idx]:
                    cover_count[t_idx] -= 1
                removed_any = True
                
        if not removed_any:
            break
            
    # 4. Local Search: Systematic Swap
    # For blocks with unique t-subsets, systematically find replacements
    # that cover all unique subsets without being already active.
    
    # Precompute block_t_indices as sets for faster membership testing
    block_t_indices_sets = [set(indices) for indices in block_t_indices]
    
    # Identify blocks with unique subsets and sort by number of unique subsets (most critical first)
    blocks_with_unique = []
    for b_idx in active_blocks:
        unique_count = sum(1 for t_idx in block_t_indices[b_idx] if cover_count[t_idx] == 1)
        if unique_count > 0:
            blocks_with_unique.append((unique_count, b_idx))
    
    # Sort by most unique subsets first (hardest to replace first)
    blocks_with_unique.sort(key=lambda x: -x[0])
    
    # For each block with unique subsets, try to find a systematic replacement
    for unique_count, b_idx in blocks_with_unique:
        if time.time() > deadline:
            break
            
        if b_idx not in active_blocks:
            continue
            
        # Get the unique subsets for this block
        unique_subsets = [t_idx for t_idx in block_t_indices[b_idx] if cover_count[t_idx] == 1]
        
        if not unique_subsets:
            continue
            
        # Find candidate blocks that cover all unique subsets
        # Use the first unique subset to get candidates, then filter
        first_t = unique_subsets[0]
        candidates = [b for b in t_subset_blocks[first_t] if b not in active_blocks]
        
        # If no candidates, try other unique subsets
        if not candidates:
            continue
        
        # Filter candidates to those that cover ALL unique subsets
        valid_candidates = []
        for cand in candidates:
            if all(t_idx in block_t_indices_sets[cand] for t_idx in unique_subsets):
                valid_candidates.append(cand)
        
        if not valid_candidates:
            continue
        
        # Among valid candidates, pick the one that covers the most other active blocks' subsets
        # (to maximize potential for further deletions)
        best_cand = None
        best_score = -1
        
        for cand in valid_candidates:
            # Score: how many t-subsets does this candidate cover that are currently covered by active blocks?
            # This indicates potential for enabling further deletions
            score = 0
            for t_idx in block_t_indices[cand]:
                if cover_count[t_idx] >= 2:
                    score += 1
            if score > best_score:
                best_score = score
                best_cand = cand
        
        if best_cand is not None:
            # Perform the swap
            active_blocks.add(best_cand)
            active_blocks.remove(b_idx)
            
            # Update cover counts
            for t_idx in block_t_indices[b_idx]:
                cover_count[t_idx] -= 1
            for t_idx in block_t_indices[best_cand]:
                cover_count[t_idx] += 1

    # Multiple deletion passes after swaps to clean up
    for _pass in range(5):
        if time.time() > deadline:
            break
        indices_to_check = list(active_blocks)
        random.shuffle(indices_to_check)
        removed_any = False
        
        for b_idx in indices_to_check:
            if b_idx not in active_blocks:
                continue
                
            can_remove = True
            for t_idx in block_t_indices[b_idx]:
                if cover_count[t_idx] <= 1:
                    can_remove = False
                    break
                    
            if can_remove:
                active_blocks.remove(b_idx)
                for t_idx in block_t_indices[b_idx]:
                    cover_count[t_idx] -= 1
                removed_any = True
        
        if not removed_any:
            break

    # Convert back to list of lists
    result = [list(all_k_subsets[b_idx]) for b_idx in active_blocks]
    
    return result
