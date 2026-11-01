import numpy as np
import itertools
import time
import random
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    # Generate all k-subsets
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    def nCr(n, r):
        if r > n: return 0
        if r == 0: return 1
        res = 1
        for i in range(r):
            res *= (n - i)
            res //= (i + 1)
        return res
            
    t_count = nCr(k, t)
    
    # Precompute block_t_indices
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = idx_map[sub]
            
    best_solution = None
    best_score = float('inf')
    
    def run_greedy(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts
        
        while remaining > 0:
            if time.time() - start_time > time_limit - 5.0:
                return None, None
            
            covered_counts = covered[block_t_indices].sum(axis=1)
            gains = t_count - covered_counts
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            if n_ks > 20:
                top_indices = np.argpartition(gains, -20)[-20:]
            else:
                top_indices = np.arange(n_ks)
                
            best_idx = top_indices[random.randint(0, len(top_indices) - 1)]
            
            blocks_indices.append(best_idx)
            
            subs_to_cover = block_t_indices[best_idx]
            prev_state = covered[subs_to_cover]
            new_covers = np.sum(~prev_state)
            remaining -= new_covers
            covered[subs_to_cover] = True
            
        current_blocks = list(blocks_indices)
        
        # Local Search Phase
        # We maintain coverage counts for the current set of blocks
        if not current_blocks:
            return current_blocks, 0
            
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
            
        improved = True
        while improved and (time.time() - start_time < time_limit - 2.0):
            improved = False
            new_blocks = []
            
            # Pass 1: Deletion
            # Shuffle to break symmetries and allow different deletion orders
            random.shuffle(current_blocks)
            for b_idx in current_blocks:
                subs_in_block = block_t_indices[b_idx]
                # If all triples in this block are covered by at least one other block
                if np.min(coverage_count[subs_in_block]) > 1:
                    coverage_count[subs_in_block] -= 1
                    improved = True
                else:
                    new_blocks.append(b_idx)
            
            current_blocks = new_blocks
            
            # If no deletion happened, try swaps to escape local minima
            if not improved and len(current_blocks) > 0:
                # Try a few random swaps
                swaps_tried = 0
                max_swaps = 50
                swap_applied = False
                
                while swaps_tried < max_swaps and not swap_applied and (time.time() - start_time < time_limit - 2.0):
                    swaps_tried += 1
                    # Pick a random block
                    b_pos = random.randint(0, len(current_blocks) - 1)
                    b_idx = current_blocks[b_pos]
                    block = all_ks[b_idx]
                    block_set = set(block)
                    
                    # Pick a point to remove
                    x = random.choice(block)
                    
                    # Pick a point to add
                    y = random.randint(0, v - 1)
                    if y in block_set:
                        continue
                    
                    # Form new block
                    new_block_list = [p for p in block if p != x]
                    new_block_list.append(y)
                    new_block_list.sort()
                    
                    # Check if this new block exists in our precomputed list
                    # We need to find the index of new_block in all_ks
                    # Since all_ks is sorted, we could use bisect, but dict lookup is faster if we have one.
                    # Let's create a mapping for block tuples to indices.
                    # Wait, we didn't create a global dict for all 346k blocks to save memory/time in precomp.
                    # Creating it now might take time.
                    # Alternative: Calculate the triples of the new block on the fly.
                    
                    new_block_triples = list(itertools.combinations(new_block_list, t))
                    new_block_triple_indices = [idx_map[ts] for ts in new_block_triples]
                    
                    old_block_triple_indices = block_t_indices[b_idx]
                    
                    # Determine lost and gained triples
                    # Lost: in old, not in new
                    # Gained: in new, not in old
                    
                    # Since sets are small (35), we can use sets of indices
                    old_set = set(old_block_triple_indices)
                    new_set = set(new_block_triple_indices)
                    
                    lost_indices = list(old_set - new_set)
                    gained_indices = list(new_set - old_set)
                    
                    # Check validity: All lost triples must have coverage > 1
                    valid = True
                    for idx in lost_indices:
                        if coverage_count[idx] <= 1:
                            valid = False
                            break
                            
                    if valid:
                        # Apply swap
                        for idx in lost_indices:
                            coverage_count[idx] -= 1
                        for idx in gained_indices:
                            coverage_count[idx] += 1
                            
                        # Update block index in current_blocks
                        # We need the index of new_block_list in all_ks.
                        # Since we don't have a direct map, and bisect on a list of tuples is O(log N),
                        # and we are doing this potentially many times, it's acceptable.
                        # However, `all_ks` is a list of tuples.
                        import bisect
                        new_block_tuple = tuple(new_block_list)
                        pos = bisect.bisect_left(all_ks, new_block_tuple)
                        if pos < n_ks and all_ks[pos] == new_block_tuple:
                            current_blocks[b_pos] = pos
                            swap_applied = True
                            improved = True # Trigger deletion pass again
                        
        return current_blocks, len(current_blocks)

    restart_count = 0
    # Run multiple restarts to find a better solution
    while time.time() - start_time < time_limit - 10.0:
        seed = restart_count
        blocks, score = run_greedy(seed)
        
        if blocks is None:
            break
            
        if score < best_score:
            best_score = score
            best_solution = blocks
            
        restart_count += 1
        
    if best_solution is None:
        blocks, score = run_greedy(0)
        if blocks is not None:
            best_solution = blocks
            best_score = score
        else:
            return []

    final_blocks = [list(all_ks[i]) for i in best_solution]
    
    return final_blocks
