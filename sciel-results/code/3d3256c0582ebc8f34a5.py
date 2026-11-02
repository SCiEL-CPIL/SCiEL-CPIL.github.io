import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin for 120s limit
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # For each block, store the indices of t-subsets it contains
    block_t_indices = []
    # For each t-subset, store which blocks contain it
    t_subset_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        t_indices = []
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            t_indices.append(idx)
            t_subset_to_blocks[idx].append(b_idx)
        block_t_indices.append(t_indices)
    
    def time_check():
        return time.time() > deadline
    
    def greedy_init():
        """Greedy construction with incremental coverage tracking."""
        counts = [0] * num_t_subsets
        blocks = []
        remaining = num_t_subsets
        
        # Track uncovered count for each block to speed up greedy
        block_uncovered = [len(block_t_indices[b]) for b in range(num_blocks)]
        
        while remaining > 0 and not time_check():
            best_block = -1
            best_cov = -1
            
            # Find block with max new coverage
            for b_idx in range(num_blocks):
                if block_uncovered[b_idx] > best_cov:
                    best_cov = block_uncovered[b_idx]
                    best_block = b_idx
                    if best_cov == remaining:
                        break
            
            if best_block == -1 or best_cov == 0:
                break
            
            blocks.append(best_block)
            
            # Update counts and block_uncovered
            for idx in block_t_indices[best_block]:
                if counts[idx] == 0:
                    remaining -= 1
                counts[idx] += 1
            
            # Update uncovered counts for all blocks
            for b_idx in range(num_blocks):
                if b_idx == best_block:
                    continue
                for idx in block_t_indices[b_idx]:
                    if counts[idx] == 1:  # Was uncovered, now covered
                        block_uncovered[b_idx] -= 1
        
        return blocks, counts
    
    def prune_redundant(blocks, counts):
        """Remove blocks where all t-subsets are covered by others."""
        new_blocks = []
        for b_idx in blocks:
            redundant = True
            for idx in block_t_indices[b_idx]:
                if counts[idx] == 1:
                    redundant = False
                    break
            if redundant:
                for idx in block_t_indices[b_idx]:
                    counts[idx] -= 1
            else:
                new_blocks.append(b_idx)
        return new_blocks
    
    def local_search(blocks, counts):
        """Iterative local search with swaps and removals."""
        improved = True
        while improved and not time_check():
            improved = False
            
            # Pass 1: Remove redundant blocks
            blocks = prune_redundant(blocks, counts)
            
            if time_check():
                break
            
            # Pass 2: Try 1-for-1 swaps
            random.shuffle(blocks)
            
            for i in range(len(blocks)):
                if time_check():
                    break
                    
                b_out = blocks[i]
                # Find t-subsets unique to b_out
                lost = [idx for idx in block_t_indices[b_out] if counts[idx] == 1]
                
                if not lost:
                    # Redundant, remove
                    for idx in block_t_indices[b_out]:
                        counts[idx] -= 1
                    blocks.pop(i)
                    improved = True
                    break
                
                # Find blocks that cover all lost subsets
                candidates = set(t_subset_to_blocks[lost[0]])
                for idx in lost[1:]:
                    candidates.intersection_update(t_subset_to_blocks[idx])
                candidates.discard(b_out)
                
                if not candidates:
                    continue
                
                # Try a random candidate
                b_in = random.choice(list(candidates))
                
                # Check if swap improves (b_in not already in blocks)
                if b_in in blocks:
                    # Just remove b_out
                    for idx in block_t_indices[b_out]:
                        counts[idx] -= 1
                    blocks.pop(i)
                    improved = True
                    break
                
                # Perform swap
                for idx in block_t_indices[b_out]:
                    counts[idx] -= 1
                blocks.pop(i)
                
                for idx in block_t_indices[b_in]:
                    counts[idx] += 1
                blocks.append(b_in)
                
                improved = True
                break
        
        return blocks, counts
    
    def try_improve(blocks, counts):
        """Try multiple local search passes with different random seeds."""
        best_blocks = blocks
        best_counts = counts.copy()
        best_len = len(blocks)
        
        for _ in range(3):  # Multiple restarts
            if time_check():
                break
            
            # Perturb: randomly remove a few blocks and re-cover
            perturbed = blocks.copy()
            p_counts = counts.copy()
            
            num_remove = max(1, len(perturbed) // 20)
            to_remove = random.sample(perturbed, min(num_remove, len(perturbed)))
            
            for b_idx in to_remove:
                if b_idx in perturbed:
                    pos = perturbed.index(b_idx)
                    perturbed.pop(pos)
                    for idx in block_t_indices[b_idx]:
                        p_counts[idx] -= 1
            
            # Re-cover with greedy
            remaining = sum(1 for c in p_counts if c == 0)
            while remaining > 0 and not time_check():
                best_b = -1
                best_cov = -1
                for b_idx in range(num_blocks):
                    cov = sum(1 for idx in block_t_indices[b_idx] if p_counts[idx] == 0)
                    if cov > best_cov:
                        best_cov = cov
                        best_b = b_idx
                        if best_cov == remaining:
                            break
                
                if best_b == -1 or best_cov == 0:
                    break
                
                perturbed.append(best_b)
                for idx in block_t_indices[best_b]:
                    if p_counts[idx] == 0:
                        remaining -= 1
                    p_counts[idx] += 1
            
            # Local search on perturbed
            perturbed, p_counts = local_search(perturbed, p_counts)
            
            if len(perturbed) < best_len:
                best_blocks = perturbed
                best_counts = p_counts
                best_len = len(perturbed)
        
        return best_blocks, best_counts
    
    # Main algorithm
    blocks, counts = greedy_init()
    blocks, counts = local_search(blocks, counts)
    blocks, counts = try_improve(blocks, counts)
    
    # Final prune
    blocks = prune_redundant(blocks, counts)
    
    # Convert indices back to lists
    result = [list(all_blocks[b_idx]) for b_idx in blocks]
    
    return result
