import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    time_limit = 120.0
    
    tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {tsub: i for i, tsub in enumerate(tsubs)}
    num_tsubs = len(tsubs)
    
    blocks_all = list(itertools.combinations(range(v), k))
    num_blocks = len(blocks_all)
    
    block_tsubs = []
    for b in blocks_all:
        covered = [tsub_to_idx[tsub] for tsub in itertools.combinations(b, t)]
        block_tsubs.append(covered)
        
    tsub_to_blocks = defaultdict(list)
    for bi in range(num_blocks):
        for ti in block_tsubs[bi]:
            tsub_to_blocks[ti].append(bi)
            
    best_blocks = None
    best_len = float('inf')
    
    num_restarts = 80
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 10.0:
            break
            
        rng = random.Random(42 + restart)
        
        uncovered = np.ones(num_tsubs, dtype=bool)
        selected_blocks = []
        used_blocks = set()
        
        while uncovered.any():
            if time.time() - start_time > time_limit - 5.0:
                break
                
            uncov_indices = np.where(uncovered)[0]
            if len(uncov_indices) == 0:
                break
            target_idx = uncov_indices[rng.randint(0, len(uncov_indices) - 1)]
            
            candidates = tsub_to_blocks[target_idx]
            
            best_block_idx = -1
            best_gain = -1
            
            for bi in candidates:
                if bi in used_blocks:
                    continue
                gain = 0
                for ti in block_tsubs[bi]:
                    if uncovered[ti]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = bi
                    if best_gain == 56:
                        break
                        
            if best_block_idx == -1:
                break
                
            selected_blocks.append(best_block_idx)
            used_blocks.add(best_block_idx)
            
            for ti in block_tsubs[best_block_idx]:
                uncovered[ti] = False
                
        if not selected_blocks:
            continue
            
        # Saturation
        cov_count = np.zeros(num_tsubs, dtype=np.int32)
        for bi in selected_blocks:
            for ti in block_tsubs[bi]:
                cov_count[ti] += 1
                
        changed = True
        while changed:
            changed = False
            if time.time() - start_time > time_limit - 5.0:
                break
            for i in range(len(selected_blocks) - 1, -1, -1):
                bi = selected_blocks[i]
                redundant = True
                for ti in block_tsubs[bi]:
                    if cov_count[ti] == 1:
                        redundant = False
                        break
                if redundant:
                    for ti in block_tsubs[bi]:
                        cov_count[ti] -= 1
                    selected_blocks.pop(i)
                    changed = True
                    
        # Local Search: Swap
        # Limit swaps to save time
        if time.time() - start_time < time_limit - 10.0:
            for _ in range(10): # 10 swap attempts
                if time.time() - start_time > time_limit - 5.0:
                    break
                if not selected_blocks:
                    break
                idx = rng.randint(0, len(selected_blocks) - 1)
                bi = selected_blocks[idx]
                
                # Find essential tsubs for bi
                essential = []
                for ti in block_tsubs[bi]:
                    if cov_count[ti] == 1:
                        essential.append(ti)
                        
                if not essential:
                    # Should have been removed, but maybe counts changed?
                    continue
                    
                # Find blocks covering all essential
                # Intersection of tsub_to_blocks for each essential tsub
                candidates = set(tsub_to_blocks[essential[0]])
                for ti in essential[1:]:
                    candidates &= set(tsub_to_blocks[ti])
                    
                # Remove self
                candidates.discard(bi)
                
                if candidates:
                    # Pick a random candidate
                    new_bi = list(candidates)[rng.randint(0, len(candidates) - 1)]
                    
                    # Check if this swap is beneficial?
                    # Just swap and re-saturate.
                    selected_blocks[idx] = new_bi
                    # Update cov_count
                    for ti in block_tsubs[bi]:
                        cov_count[ti] -= 1
                    for ti in block_tsubs[new_bi]:
                        cov_count[ti] += 1
                        
                    # Re-saturate
                    changed = True
                    while changed:
                        changed = False
                        if time.time() - start_time > time_limit - 5.0:
                            break
                        for j in range(len(selected_blocks) - 1, -1, -1):
                            bj = selected_blocks[j]
                            redundant = True
                            for ti in block_tsubs[bj]:
                                if cov_count[ti] == 1:
                                    redundant = False
                                    break
                            if redundant:
                                for ti in block_tsubs[bj]:
                                    cov_count[ti] -= 1
                                selected_blocks.pop(j)
                                changed = True
                                
        if len(selected_blocks) < best_len:
            best_len = len(selected_blocks)
            best_blocks = [blocks_all[bi] for bi in selected_blocks]
            
    if best_blocks is None:
        return []
        
    return [list(b) for b in best_blocks]
