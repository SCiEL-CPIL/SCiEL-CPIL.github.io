import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precomputation
    # 1. Map all t-subsets to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # 2. Precompute for each k-block, the indices of the t-subsets it covers
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    block_t_indices = [] # list of arrays, each of length C(k,t)
    t_subset_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for b_idx, block in enumerate(all_k_blocks):
        t_subs_in_block = list(itertools.combinations(block, t))
        indices = np.array([t_subset_to_idx[s] for s in t_subs_in_block], dtype=np.int32)
        block_t_indices.append(indices)
        
        for idx in indices:
            t_subset_to_blocks[idx].append(b_idx)

    best_blocks = None
    best_score = float('inf')
    
    # Helper to reduce blocks
    def reduce_blocks(blocks_idx, coverage):
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b_idx in blocks_idx:
                idxs = block_t_indices[b_idx]
                # Check if any t-subset is uniquely covered by this block
                if not np.any(coverage[idxs] == 1):
                    # Safe to remove
                    coverage[idxs] -= 1
                    changed = True
                else:
                    new_blocks.append(b_idx)
            blocks_idx = new_blocks
        return blocks_idx

    # Multiple restarts
    for restart in range(50):
        if time.time() > deadline - 10.0:
            break
            
        random.seed(restart)
        
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        uncovered_indices = list(range(num_t_subsets))
        blocks_idx = []
        
        # Greedy Phase
        while uncovered_indices:
            if time.time() > deadline - 5.0:
                break
            
            # Pick a random uncovered subset
            pos = random.randint(0, len(uncovered_indices) - 1)
            u_idx = uncovered_indices[pos]
            
            # Get candidates
            candidates = t_subset_to_blocks[u_idx]
            
            # Find best candidate (max gain)
            best_gain = -1
            best_block_idx = -1
            
            for b_idx in candidates:
                gain = np.count_nonzero(coverage[block_t_indices[b_idx]] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            # Add best block
            blocks_idx.append(best_block_idx)
            
            # Update coverage
            idxs = block_t_indices[best_block_idx]
            coverage[idxs] += 1
            
            # Update uncovered list
            uncovered_indices = [x for x in uncovered_indices if coverage[x] == 0]
        
        if len(uncovered_indices) > 0:
            continue
            
        # Reduction Phase
        blocks_idx = reduce_blocks(blocks_idx, coverage)
        
        current_score = len(blocks_idx)
        if current_score < best_score:
            best_score = current_score
            best_blocks = [list(all_k_blocks[i]) for i in blocks_idx]
            
            # Local Search: Add-Reduce
            ls_start = time.time()
            while time.time() < ls_start + 2.0:
                b_candidate = random.randint(0, num_k_blocks - 1)
                if b_candidate in blocks_idx:
                    continue
                
                old_blocks = list(blocks_idx)
                old_coverage = coverage.copy()
                
                blocks_idx.append(b_candidate)
                coverage[block_t_indices[b_candidate]] += 1
                
                new_blocks = reduce_blocks(blocks_idx, coverage)
                
                if len(new_blocks) < len(old_blocks):
                    blocks_idx = new_blocks
                    best_score = len(blocks_idx)
                    best_blocks = [list(all_k_blocks[i]) for i in blocks_idx]
                else:
                    blocks_idx = old_blocks
                    coverage = old_coverage

    return best_blocks
