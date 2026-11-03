"""Covering design C(v,k,t) using WalkSAT-style stochastic local search on block assignments."""
import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112  # safety margin before 150s kill

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)

    # Precompute all possible k-subsets (blocks)
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

    def walk_sat_search(seed, max_steps=50000, target_size=None):
        """
        WalkSAT-style stochastic local search.
        State: a set of selected blocks (represented as a boolean array over all_blocks).
        Objective: minimize the number of uncovered t-subsets.
        Move: randomly pick an uncovered t-subset, then either add a block covering it 
              or remove a random block (to allow escaping local minima).
        """
        rng = random.Random(seed)
        
        # Start with a random set of blocks, slightly larger than expected optimum
        if target_size is None:
            target_size = int(n_tsubs / (k * (k-1) * (k-2) / (t*(t-1)*(t-2))) * 1.5) + 20
        
        # Ensure target_size is reasonable
        target_size = max(50, min(target_size, n_blocks))
        
        # Initialize with random blocks
        selected = np.zeros(n_blocks, dtype=bool)
        cov_count = np.zeros(n_tsubs, dtype=np.int32)
        
        # Add initial random blocks
        initial_blocks = rng.sample(range(n_blocks), min(target_size, n_blocks))
        for bi in initial_blocks:
            selected[bi] = True
            for idx in block_to_tsubs[bi]:
                cov_count[idx] += 1
        
        # Track best solution found
        best_count = np.sum(~selected)
        best_selected = selected.copy()
        best_cov = cov_count.copy()
        
        # WalkSAT loop
        for step in range(max_steps):
            if time.time() > deadline - 15:
                break
            
            # Count uncovered t-subsets
            uncovered = cov_count == 0
            n_uncovered = np.sum(uncovered)
            
            # Update best if we have a valid cover
            if n_uncovered == 0:
                current_size = np.sum(selected)
                if current_size < best_count:
                    best_count = current_size
                    best_selected = selected.copy()
                    best_cov = cov_count.copy()
            
            # If no uncovered, try to remove random blocks to shrink
            if n_uncovered == 0:
                if rng.random() < 0.7:
                    # Try to remove a random selected block
                    selected_indices = np.where(selected)[0]
                    if len(selected_indices) > 0:
                        bi = rng.choice(selected_indices.tolist())
                        selected[bi] = False
                        for idx in block_to_tsubs[bi]:
                            cov_count[idx] -= 1
                        # Check if still valid
                        if np.any(cov_count == 0):
                            # Restore
                            selected[bi] = True
                            for idx in block_to_tsubs[bi]:
                                cov_count[idx] += 1
                continue
            
            # Pick a random uncovered t-subset
            uncovered_indices = np.where(uncovered)[0]
            ts_idx = rng.choice(uncovered_indices.tolist())
            
            # Get blocks that cover this t-subset
            covering_blocks = tsub_to_blocks[ts_idx]
            if not covering_blocks:
                continue
            
            # With some probability, add a block covering the uncovered t-subset
            if rng.random() < 0.6:
                # Pick a random block covering ts_idx
                bi = rng.choice(covering_blocks)
                if not selected[bi]:
                    selected[bi] = True
                    for idx in block_to_tsubs[bi]:
                        cov_count[idx] += 1
            else:
                # With some probability, remove a random selected block (to escape local minima)
                selected_indices = np.where(selected)[0]
                if len(selected_indices) > 0:
                    bi = rng.choice(selected_indices.tolist())
                    selected[bi] = False
                    for idx in block_to_tsubs[bi]:
                        cov_count[idx] -= 1
        
        # Return the best valid cover found
        if np.any(best_cov == 0):
            # No valid cover found, return empty
            return []
        
        # Extract blocks from best_selected
        blocks = []
        for bi in np.where(best_selected)[0]:
            blocks.append(all_blocks[bi])
        return blocks

    def greedy_build(seed):
        """Fast greedy set cover to get a feasible initial solution."""
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubs, dtype=np.int32)
        blocks_used = []
        used_block_indices = set()
        
        while True:
            if uncovered.sum() == 0:
                break
            
            uncovered_indices = np.where(uncovered == 1)[0]
            if len(uncovered_indices) == 0:
                break
            
            best_block_idx = -1
            best_gain = -1
            best_block_tsubs = []
            
            # Sample constrained t-subsets for efficiency
            sample_size = min(30, len(uncovered_indices))
            sample_idx = rng.sample(uncovered_indices.tolist(), sample_size)
            
            for ts_idx in sample_idx:
                covering_blocks = [bi for bi in tsub_to_blocks[ts_idx] if bi not in used_block_indices]
                if not covering_blocks:
                    continue
                
                rng.shuffle(covering_blocks)
                for bi in covering_blocks[:15]:
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
            
            if time.time() > deadline - 30:
                break
        
        return blocks_used

    def reduce_blocks(blocks, max_passes=3):
        """Remove redundant blocks."""
        if not blocks:
            return blocks
        
        blocks_list = list(blocks)
        
        for _ in range(max_passes):
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
    
    # Phase 1: Get a feasible cover via greedy
    greedy_blocks = greedy_build(42)
    if greedy_blocks:
        reduced_greedy = reduce_blocks(greedy_blocks, max_passes=2)
        if len(reduced_greedy) < best_count:
            best_blocks = reduced_greedy
            best_count = len(reduced_greedy)
    
    # Phase 2: WalkSAT-style stochastic local search with multiple restarts
    for seed in range(20):
        if time.time() > deadline - 20:
            break
        
        blocks = walk_sat_search(seed, max_steps=30000)
        if blocks:
            reduced = reduce_blocks(blocks, max_passes=2)
            if len(reduced) < best_count:
                best_blocks = reduced
                best_count = len(reduced)
    
    # Phase 3: Final aggressive reduction on best solution
    if best_blocks is not None and time.time() < deadline - 5:
        best_blocks = reduce_blocks(best_blocks, max_passes=5)
    
    return best_blocks if best_blocks else []
