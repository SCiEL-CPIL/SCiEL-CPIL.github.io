"""Covering design C(v,k,t) with efficient greedy + aggressive reduction + local search."""
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
    
    def greedy_build(seed, use_random_order=True):
        """Randomized greedy covering with multiple strategies."""
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubs, dtype=np.int32)
        blocks_used = []
        used_block_indices = set()
        
        while True:
            # Check if all covered
            if uncovered.sum() == 0:
                break
            
            # Find uncovered t-subsets
            uncovered_indices = np.where(uncovered == 1)[0]
            if len(uncovered_indices) == 0:
                break
            
            best_block_idx = -1
            best_gain = -1
            best_block_tsubs = []
            
            # Strategy 1: Pick most constrained uncovered t-subset (fewest covering blocks)
            # then pick best block covering it
            if use_random_order and len(uncovered_indices) > 50:
                # Sample constrained t-subsets
                # Compute how many blocks cover each uncovered t-subset
                constrained_score = []
                for ts_idx in uncovered_indices:
                    n_covering = len(tsub_to_blocks[ts_idx])
                    # Filter to unused blocks
                    unused_count = sum(1 for bi in tsub_to_blocks[ts_idx] if bi not in used_block_indices)
                    if unused_count > 0:
                        constrained_score.append((unused_count, ts_idx))
                
                if not constrained_score:
                    break
                
                # Pick a few of the most constrained
                constrained_score.sort(key=lambda x: x[0])
                sample_size = min(20, len(constrained_score))
                sample = constrained_score[:sample_size]
                
                for _, ts_idx in sample:
                    covering_blocks = [bi for bi in tsub_to_blocks[ts_idx] if bi not in used_block_indices]
                    if not covering_blocks:
                        continue
                    
                    # Shuffle and try top candidates
                    rng.shuffle(covering_blocks)
                    for bi in covering_blocks[:5]:
                        block_tsubs = block_to_tsubs[bi]
                        gain = sum(1 for idx in block_tsubs if uncovered[idx] == 1)
                        if gain > best_gain:
                            best_gain = gain
                            best_block_idx = bi
                            best_block_tsubs = block_tsubs
            else:
                # Simpler approach: sample uncovered t-subsets
                sample_size = min(30, len(uncovered_indices))
                sample_idx = rng.sample(uncovered_indices.tolist(), sample_size)
                
                for ts_idx in sample_idx:
                    covering_blocks = [bi for bi in tsub_to_blocks[ts_idx] if bi not in used_block_indices]
                    if not covering_blocks:
                        continue
                    
                    rng.shuffle(covering_blocks)
                    for bi in covering_blocks[:8]:
                        block_tsubs = block_to_tsubs[bi]
                        gain = sum(1 for idx in block_tsubs if uncovered[idx] == 1)
                        if gain > best_gain:
                            best_gain = gain
                            best_block_idx = bi
                            best_block_tsubs = block_tsubs
            
            if best_block_idx == -1:
                # Fallback: try all blocks for the first uncovered t-subset
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
            
            # Time check
            if time.time() > deadline - 15:
                break
        
        return blocks_used
    
    def reduce_blocks(blocks, max_passes=3):
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
                # Try removing blocks in random order for more diverse reductions
                order = list(range(len(blocks_list)))
                random.shuffle(order)
                
                for i in order:
                    if i >= len(blocks_list):
                        continue
                    # Check if block i is redundant
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
    
    def try_remove_and_replace(blocks, rng, max_attempts=5):
        """Try to remove a block and replace it with a better one."""
        if not blocks:
            return blocks
        
        blocks_list = list(blocks)
        n = len(blocks_list)
        
        # Build initial coverage
        cov_count = np.zeros(n_tsubs, dtype=np.int32)
        block_tsubs_list = []
        for block in blocks_list:
            indices = get_block_tsub_indices(tuple(block))
            block_tsubs_list.append(indices)
            for idx in indices:
                cov_count[idx] += 1
        
        for _ in range(max_attempts):
            # Pick a random block to try removing
            i = rng.randrange(n)
            if i >= len(blocks_list):
                continue
            
            saved_block = blocks_list[i]
            saved_tsubs = block_tsubs_list[i]
            
            # Remove it
            blocks_list.pop(i)
            block_tsubs_list.pop(i)
            for idx in saved_tsubs:
                cov_count[idx] -= 1
            
            # Check if still valid
            if cov_count.min() >= 1:
                # Valid! We reduced the size
                return blocks_list
            
            # Not valid, need to add a block back
            # Find which t-subsets are now uncovered
            uncovered_indices = np.where(cov_count == 0)[0]
            
            # Try to find a block that covers the most uncovered t-subsets
            best_add = -1
            best_gain = 0
            
            for ts_idx in uncovered_indices[:10]:
                candidates = tsub_to_blocks[ts_idx]
                for cand in candidates:
                    if cand in [blocks_list[j] for j in range(len(blocks_list))]:
                        continue
                    cand_tsubs = block_to_tsubs[cand]
                    gain = sum(1 for idx in cand_tsubs if cov_count[idx] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_add = cand
            
            if best_add != -1 and best_gain > 0:
                # Add the new block
                blocks_list.append(best_add)
                new_tsubs = block_to_tsubs[best_add]
                for idx in new_tsubs:
                    cov_count[idx] += 1
                block_tsubs_list.append(new_tsubs)
            else:
                # Restore
                blocks_list.insert(i, saved_block)
                block_tsubs_list.insert(i, saved_tsubs)
                for idx in saved_tsubs:
                    cov_count[idx] += 1
        
        return blocks_list
    
    # Main search loop
    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Multiple greedy constructions with different seeds
    for seed in range(20):
        if time.time() > deadline - 25:
            break
        
        blocks = greedy_build(seed)
        reduced = reduce_blocks(blocks, max_passes=2)
        
        if len(reduced) < best_count:
            best_blocks = reduced
            best_count = len(reduced)
    
    # Phase 2: Local search on best solution
    if best_blocks is not None and time.time() < deadline - 15:
        rng = random.Random(12345)
        current = list(best_blocks)
        
        for iteration in range(50):
            if time.time() > deadline - 10:
                break
            
            # Try remove-and-replace
            improved = try_remove_and_replace(current, rng, max_attempts=10)
            if len(improved) < len(current):
                current = improved
                if len(current) < best_count:
                    best_blocks = current
                    best_count = len(current)
            
            # Try removing one block and checking validity
            if len(current) > 0:
                n = len(current)
                for i in range(n):
                    if time.time() > deadline - 10:
                        break
                    saved = current[i]
                    current.pop(i)
                    
                    # Quick validity check
                    cov = np.zeros(n_tsubs, dtype=np.int32)
                    for block in current:
                        for idx in get_block_tsub_indices(tuple(block)):
                            cov[idx] += 1
                    
                    if cov.min() >= 1:
                        if len(current) < best_count:
                            best_blocks = current
                            best_count = len(current)
                    else:
                        current.insert(i, saved)
            
            # Final reduction
            if time.time() < deadline - 5:
                current = reduce_blocks(current, max_passes=1)
                if len(current) < best_count:
                    best_blocks = current
                    best_count = len(current)
    
    # Final aggressive reduction
    if best_blocks is not None and time.time() < deadline - 2:
        best_blocks = reduce_blocks(best_blocks, max_passes=5)
    
    return best_blocks if best_blocks else greedy_build(0)
