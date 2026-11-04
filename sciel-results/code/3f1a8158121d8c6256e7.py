"""Covering C(19,8,5) via maximum-coverage pruning from all blocks."""
import numpy as np
import itertools
import sys

def construct(v, k, t):
    # Precompute all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_index = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    # Precompute all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    n_k = len(k_blocks)
    
    # For each block, store the list of t-subset indices it covers
    block_t_list = []
    for block in k_blocks:
        covered = [t_index[ts] for ts in itertools.combinations(block, t)]
        block_t_list.append(covered)
        
    # For each t-subset, store the list of block indices that cover it
    # This is needed to update block_essential_count efficiently
    t_to_blocks = [[] for _ in range(n_t)]
    for b_idx, covered in enumerate(block_t_list):
        for t_idx in covered:
            t_to_blocks[t_idx].append(b_idx)
            
    # Coverage counts: how many remaining blocks cover each t-subset
    coverage = np.zeros(n_t, dtype=np.int32)
    for covered in block_t_list:
        for idx in covered:
            coverage[idx] += 1
    
    # Active flags for each block
    active = np.ones(n_k, dtype=bool)
    
    # block_essential_count[b] = number of t-subsets in block b that have coverage == 1
    # If this count is 0, the block is removable.
    block_essential_count = np.zeros(n_k, dtype=np.int32)
    
    # Initialize block_essential_count
    # A t-subset is "essential" if its coverage is 1.
    # For each such t-subset, all blocks covering it get +1 to their essential count.
    for t_idx in range(n_t):
        if coverage[t_idx] == 1:
            for b_idx in t_to_blocks[t_idx]:
                block_essential_count[b_idx] += 1
                
    # Stack of blocks that are potentially removable (essential count == 0)
    # We use a list as a stack. We only add blocks initially.
    # As we remove blocks, coverage drops, making more t-subsets essential (count 1),
    # which increases block_essential_count for other blocks.
    # Thus, blocks that are NOT removable initially will NEVER become removable later.
    # Blocks that ARE removable initially might become non-removable if their
    # coverage of some t-subset drops to 1 (making that t-subset essential).
    # So we just need to process the initial set of removable blocks.
    
    removable_stack = [i for i in range(n_k) if block_essential_count[i] == 0]
    
    # Process removals
    while removable_stack:
        b_idx = removable_stack.pop()
        
        # Check if it's still removable
        # It is removable if it is active AND its essential count is still 0.
        # (It might have been marked as non-removable by previous removals)
        if not active[b_idx]:
            continue
        if block_essential_count[b_idx] > 0:
            continue
            
        # Remove this block
        active[b_idx] = False
        
        # Update coverage and essential counts
        for t_idx in block_t_list[b_idx]:
            old_cov = coverage[t_idx]
            coverage[t_idx] -= 1
            
            # If coverage drops to 1, this t-subset becomes essential.
            # This means all *other* active blocks covering it must be kept.
            # So we increment their block_essential_count.
            if coverage[t_idx] == 1:
                for other_b in t_to_blocks[t_idx]:
                    if active[other_b]:
                        block_essential_count[other_b] += 1
                        
    # Collect remaining blocks
    result = [list(k_blocks[i]) for i in range(n_k) if active[i]]
    
    # Verify coverage (optional, for safety)
    # Every t-subset must be covered by at least one block
    if n_t > 0:
        if np.any(coverage < 1):
            # This shouldn't happen if our logic is correct, but as a safeguard:
            # Fall back to greedy on uncovered t-subsets
            uncovered = set()
            for idx in range(n_t):
                if coverage[idx] < 1:
                    uncovered.add(t_subsets[idx])
            
            while uncovered:
                # Greedy: find block covering most uncovered
                best_block = None
                best_gain = -1
                for i in range(n_k):
                    if active[i]:
                        covered = block_t_list[i]
                        gain = sum(1 for idx in covered if t_subsets[idx] in uncovered)
                    else:
                        covered = block_t_list[i]
                        gain = sum(1 for idx in covered if t_subsets[idx] in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_block = i
                if best_block is None:
                    break
                active[best_block] = True
                result.append(list(k_blocks[best_block]))
                for idx in block_t_list[best_block]:
                    s = t_subsets[idx]
                    if s in uncovered:
                        uncovered.discard(s)
                    coverage[idx] += 1
            result.sort()
    
    return result
