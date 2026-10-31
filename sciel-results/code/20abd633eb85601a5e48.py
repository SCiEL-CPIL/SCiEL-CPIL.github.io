"""Combined approach: Global Greedy Set Cover with vectorized NumPy scoring + local search minimization."""
import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115  # Leave margin for the 120s limit
    
    all_points = list(range(v))
    all_tsubsets = list(itertools.combinations(all_points, t))
    tsubset_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    n_tsubsets = len(all_tsubsets)
    
    # Precompute all k-blocks and their covered t-subset indices
    all_blocks = list(itertools.combinations(all_points, k))
    n_blocks = len(all_blocks)
    
    # Build block_to_tsubs: list of t-subset indices covered by each block
    block_to_tsubs = []
    # Build tsub_to_blocks: list of block indices covering each t-subset
    tsub_to_blocks = [[] for _ in range(n_tsubsets)]
    
    for b in all_blocks:
        covered = []
        for ts in itertools.combinations(b, t):
            idx = tsubset_index[ts]
            covered.append(idx)
            tsub_to_blocks[idx].append(len(block_to_tsubs))
        block_to_tsubs.append(covered)
    
    # Convert to numpy arrays for fast operations
    # block_cover_matrix[i][j] = 1 if block i covers t-subset j
    # This might be too large; instead use sparse representation
    # block_to_tsubs is a list of lists of ints
    
    def greedy_init():
        """Global greedy set cover using numpy for fast gain computation."""
        uncovered = np.ones(n_tsubsets, dtype=np.int8)
        n_uncovered = n_tsubsets
        blocks = []
        used_blocks = set()
        
        while n_uncovered > 0:
            if (time.time() - start_time) > time_limit:
                break
            
            best_block = None
            best_gain = -1
            
            for bi, b in enumerate(all_blocks):
                if bi in used_blocks:
                    continue
                # Count how many uncovered t-subsets this block covers
                gain = 0
                for ts_idx in block_to_tsubs[bi]:
                    if uncovered[ts_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
            
            if best_block is None or best_gain == 0:
                break
            
            blocks.append(best_block)
            used_blocks.add(best_block)
            
            for ts_idx in block_to_tsubs[best_block]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = 0
                    n_uncovered -= 1
                    
        return [all_blocks[bi] for bi in blocks]
    
    def local_search(blocks):
        """Local search: remove redundants, then try 1-exchanges."""
        current_blocks = list(blocks)
        current_block_set = set(current_blocks)
        
        # Coverage counts
        coverage = [0] * n_tsubsets
        for b in current_blocks:
            for ts_idx in block_to_tsubs[list(all_blocks).index(b)]:
                coverage[ts_idx] += 1
        
        def is_redundant(b):
            bi = list(all_blocks).index(b)
            for ts_idx in block_to_tsubs[bi]:
                if coverage[ts_idx] == 1:
                    return False
            return True
        
        def remove_block(b):
            bi = list(all_blocks).index(b)
            current_block_set.remove(b)
            current_blocks.remove(b)
            for ts_idx in block_to_tsubs[bi]:
                coverage[ts_idx] -= 1
        
        def add_block(b):
            bi = list(all_blocks).index(b)
            current_block_set.add(b)
            current_blocks.append(b)
            for ts_idx in block_to_tsubs[bi]:
                coverage[ts_idx] += 1
        
        # Phase 1: Remove redundants
        changed = True
        while changed:
            changed = False
            candidates = list(current_blocks)
            for b in candidates:
                if b not in current_block_set:
                    continue
                if is_redundant(b):
                    remove_block(b)
                    changed = True
                    if (time.time() - start_time) > time_limit:
                        return current_blocks
        
        # Phase 2: 1-Exchange (Swap)
        # Try to swap each block with a better one
        for _ in range(5):  # Repeat passes
            if (time.time() - start_time) > time_limit:
                break
            improved = False
            blocks_to_try = current_blocks[:]
            random.shuffle(blocks_to_try)
            
            for b in blocks_to_try:
                if b not in current_block_set:
                    continue
                
                bi = list(all_blocks).index(b)
                
                # Find critical t-subsets for b
                critical_ts = []
                for ts_idx in block_to_tsubs[bi]:
                    if coverage[ts_idx] == 1:
                        critical_ts.append(ts_idx)
                
                if not critical_ts:
                    # b is redundant now
                    remove_block(b)
                    improved = True
                    continue
                else:
                    # Find a candidate b_new that covers all critical_ts
                    candidates = None
                    for ts_idx in critical_ts:
                        blocks_covering = tsub_to_blocks[ts_idx]
                        if candidates is None:
                            candidates = set(blocks_covering)
                        else:
                            candidates.intersection_update(blocks_covering)
                    
                    if candidates:
                        candidates -= current_block_set
                        if candidates:
                            # Try swapping
                            for b_new_idx in list(candidates)[:10]:  # Limit candidates
                                b_new = all_blocks[b_new_idx]
                                # Check if swap improves
                                # Remove b, add b_new, then remove redundants
                                remove_block(b)
                                add_block(b_new)
                                
                                # Try to remove redundants
                                changed = True
                                while changed:
                                    changed = False
                                    candidates2 = list(current_blocks)
                                    for b2 in candidates2:
                                        if b2 not in current_block_set:
                                            continue
                                        if is_redundant(b2):
                                            remove_block(b2)
                                            changed = True
                                
                                # If we ended up with fewer blocks, keep it
                                # Otherwise revert
                                if len(current_blocks) >= len(blocks_to_try):
                                    # Revert
                                    remove_block(b_new)
                                    add_block(b)
                                else:
                                    improved = True
                                    break
                            if improved:
                                break
                
                if (time.time() - start_time) > time_limit:
                    break
        
        return current_blocks
    
    # Run greedy init
    blocks = greedy_init()
    
    # Run local search
    if (time.time() - start_time) < time_limit:
        blocks = local_search(blocks)
    
    return blocks
