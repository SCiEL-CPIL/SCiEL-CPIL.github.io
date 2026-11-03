"""Covering design C(v,k,t) using Column Generation (Greedy Set Cover master + pricing) with aggressive reduction."""
import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)

    # Precompute all possible k-subsets (blocks) and their covered t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, precompute the indices of t-subsets it covers
    block_to_tsubs = []
    for block in all_blocks:
        indices = []
        for ts in itertools.combinations(block, t):
            idx = tsub_index.get(ts)
            if idx is not None:
                indices.append(idx)
        block_to_tsubs.append(indices)

    # Reverse index: for each t-subset, which blocks cover it
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    for bi, indices in enumerate(block_to_tsubs):
        for idx in indices:
            tsub_to_blocks[idx].append(bi)

    def get_block_tsub_indices(block_tuple):
        """Return list of t-subset indices covered by a block (as tuple)."""
        indices = []
        for ts in itertools.combinations(block_tuple, t):
            idx = tsub_index.get(ts)
            if idx is not None:
                indices.append(idx)
        return indices

    def column_generation_cover(seed, max_iterations=200):
        """
        Column generation framework for minimum weight set cover.
        Master problem: minimize sum of selected blocks s.t. every t-subset covered.
        Pricing subproblem: find a block with max reduced cost (uncovered coverage per unit weight).
        We use a greedy heuristic for pricing and iteratively add columns.
        """
        rng = random.Random(seed)
        
        # Start with a feasible solution from greedy
        uncovered = np.ones(n_tsubs, dtype=np.int32)
        blocks_used = []
        used_block_indices = set()
        
        # Initial greedy phase to get a feasible cover
        while True:
            if uncovered.sum() == 0:
                break
            
            uncovered_indices = np.where(uncovered == 1)[0]
            if len(uncovered_indices) == 0:
                break
            
            best_block_idx = -1
            best_gain = -1
            best_block_tsubs = []
            
            # Sample constrained t-subsets
            sample_size = min(50, len(uncovered_indices))
            sample_idx = rng.sample(uncovered_indices.tolist(), sample_size)
            
            for ts_idx in sample_idx:
                covering_blocks = [bi for bi in tsub_to_blocks[ts_idx] if bi not in used_block_indices]
                if not covering_blocks:
                    continue
                
                rng.shuffle(covering_blocks)
                for bi in covering_blocks[:10]:
                    block_tsubs = block_to_tsubs[bi]
                    gain = sum(1 for idx in block_tsubs if uncovered[idx] == 1)
                    if gain > best_gain:
                        best_gain = gain
                        best_block_idx = bi
                        best_block_tsubs = block_tsubs
            
            if best_block_idx == -1:
                # Fallback
                ts_idx = uncovered_indices[0]
                for bi in tsub_to_blocks[ts_idx]:
                    if bi in used_block_indices:
                        continue
                    block_tsubs = block_to_tsubs[bi]
                    gain = sum(1 for idx in block_tsubs if uncovered[idx] == 1)
                    if gain > best_gain:
                        best_gain = gain
                        best_block_idx = bi
                        best_block_tsubs = block_tsubs
            
            if best_block_idx == -1:
                break
            
            used_block_indices.add(best_block_idx)
            blocks_used.append(all_blocks[best_block_idx])
            
            for idx in best_block_tsubs:
                uncovered[idx] = 0
            
            if time.time() > deadline - 20:
                break
        
        if not blocks_used:
            return []
        
        # Now enter column generation / local search phase
        # Maintain coverage counts
        cov_count = np.zeros(n_tsubs, dtype=np.int32)
        block_tsubs_list = []
        for block in blocks_used:
            indices = get_block_tsub_indices(tuple(block))
            block_tsubs_list.append(indices)
            for idx in indices:
                cov_count[idx] += 1
        
        # Column generation: iteratively try to remove blocks and replace with better ones
        for iteration in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            # Try to remove each block and see if we can find a replacement that covers more
            n = len(blocks_used)
            if n == 0:
                break
            
            # Random order for removal attempts
            order = list(range(n))
            rng.shuffle(order)
            
            for i in order:
                if i >= len(blocks_used):
                    continue
                
                saved_block = blocks_used[i]
                saved_tsubs = block_tsubs_list[i]
                
                # Temporarily remove
                blocks_used.pop(i)
                block_tsubs_list.pop(i)
                for idx in saved_tsubs:
                    cov_count[idx] -= 1
                
                # Check if still valid
                if cov_count.min() >= 1:
                    # Valid! We reduced the size. Keep it removed.
                    continue
                
                # Not valid. Find uncovered t-subsets
                uncovered_indices = np.where(cov_count == 0)[0]
                
                # Pricing: find a block that covers the most uncovered t-subsets
                best_add = -1
                best_gain = 0
                
                # Sample uncovered t-subsets for pricing
                sample_size = min(20, len(uncovered_indices))
                sample = rng.sample(uncovered_indices.tolist(), sample_size)
                
                for ts_idx in sample:
                    candidates = tsub_to_blocks[ts_idx]
                    for cand in candidates:
                        # Check if this block is already in the current set
                        if cand in [blocks_used[j] for j in range(len(blocks_used))]:
                            continue
                        cand_tsubs = block_to_tsubs[cand]
                        gain = sum(1 for idx in cand_tsubs if cov_count[idx] == 0)
                        if gain > best_gain:
                            best_gain = gain
                            best_add = cand
                
                if best_add != -1 and best_gain > 0:
                    # Add the new block
                    blocks_used.append(all_blocks[best_add])
                    new_tsubs = block_to_tsubs[best_add]
                    for idx in new_tsubs:
                        cov_count[idx] += 1
                    block_tsubs_list.append(new_tsubs)
                else:
                    # Restore original block
                    blocks_used.insert(i, saved_block)
                    block_tsubs_list.insert(i, saved_tsubs)
                    for idx in saved_tsubs:
                        cov_count[idx] += 1
        
        return blocks_used

    def reduce_blocks(blocks, max_passes=5):
        """Aggressively remove redundant blocks."""
        if not blocks:
            return blocks
        
        blocks_list = list(blocks)
        
        for _ in range(max_passes):
            # Build coverage count
            cov_count = np.zeros(n_tsubs, dtype=np.int32)
            block_tsubs_list = []
            for block in blocks_list:
                indices = get_block_tsub_indices(tuple(block))
                block_tsubs_list.append(indices)
                for idx in indices:
                    cov_count[idx] += 1
            
            improved = True
            while improved:
                improved = False
                order = list(range(len(blocks_list)))
                random.shuffle(order)
                
                for i in order:
                    if i >= len(blocks_list):
                        continue
                    redundant = True
                    for idx in block_tsubs_list[i]:
                        if cov_count[idx] <= 1:
                            redundant = False
                            break
                    
                    if redundant:
                        for idx in block_tsubs_list[i]:
                            cov_count[idx] -= 1
                        blocks_list.pop(i)
                        block_tsubs_list.pop(i)
                        improved = True
            
            if not improved:
                break
        
        return blocks_list

    # Main search loop
    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Multiple column generation constructions with different seeds
    for seed in range(15):
        if time.time() > deadline - 30:
            break
        
        blocks = column_generation_cover(seed)
        reduced = reduce_blocks(blocks, max_passes=3)
        
        if len(reduced) < best_count:
            best_blocks = reduced
            best_count = len(reduced)
    
    # Phase 2: Final aggressive reduction on best solution
    if best_blocks is not None and time.time() < deadline - 5:
        best_blocks = reduce_blocks(best_blocks, max_passes=5)
    
    return best_blocks if best_blocks else []
