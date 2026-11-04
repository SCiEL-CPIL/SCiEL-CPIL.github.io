import itertools
import time
import random
import copy
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    timeout = 145  # Safety margin
    
    points = range(v)
    
    # Generate all k-blocks
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    # Generate all t-subsets
    all_subsets = list(itertools.combinations(points, t))
    N_subsets = len(all_subsets)
    
    # Map subset to index
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # Precompute for each block:
    # 1. The list of subset indices it covers
    # 2. A bitmask (big int) representing these subsets
    block_subsets = []
    block_masks = []
    
    for b in all_blocks:
        subs = [subset_to_idx[s] for s in itertools.combinations(b, t)]
        block_subsets.append(subs)
        mask = 0
        for s_idx in subs:
            mask |= (1 << s_idx)
        block_masks.append(mask)
        
    # Precompute for each subset: list of blocks that cover it
    blocks_covering_subset = [[] for _ in range(N_subsets)]
    for i, subs in enumerate(block_subsets):
        for s in subs:
            blocks_covering_subset[s].append(i)
            
    def run_greedy():
        cover_count = np.zeros(N_subsets, dtype=np.int32)
        uncovered_mask = (1 << N_subsets) - 1
        
        selected_blocks = []
        
        while uncovered_mask != 0:
            if time.time() - start_time > timeout:
                return None, None
                
            best_idx = -1
            max_cleared = 0
            
            for i in range(N_blocks):
                cleared = (uncovered_mask & block_masks[i]).bit_count()
                if cleared > max_cleared:
                    max_cleared = cleared
                    best_idx = i
            
            if max_cleared == 0:
                break
                
            selected_blocks.append(best_idx)
            
            for s in block_subsets[best_idx]:
                cover_count[s] += 1
                if cover_count[s] == 1:
                    uncovered_mask &= ~(1 << s)
                    
        # Redundancy Removal
        changed = True
        while changed:
            changed = False
            for i in range(len(selected_blocks) - 1, -1, -1):
                b_idx = selected_blocks[i]
                is_redundant = True
                for s in block_subsets[b_idx]:
                    if cover_count[s] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for s in block_subsets[b_idx]:
                        cover_count[s] -= 1
                    selected_blocks.pop(i)
                    changed = True
                    
        return selected_blocks, cover_count

    def local_search_1swap(selected_blocks, cover_count):
        # Deep copy to avoid modifying original
        selected_blocks = list(selected_blocks)
        cover_count = cover_count.copy()
        
        max_swaps = 1000
        swaps_done = 0
        
        while swaps_done < max_swaps and (time.time() - start_time < timeout - 5):
            improved = False
            
            indices = list(range(len(selected_blocks)))
            random.shuffle(indices)
            
            for i in indices:
                if i >= len(selected_blocks):
                    continue
                b_idx = selected_blocks[i]
                
                critical_subsets = []
                for s in block_subsets[b_idx]:
                    if cover_count[s] == 1:
                        critical_subsets.append(s)
                
                if not critical_subsets:
                    for s in block_subsets[b_idx]:
                        cover_count[s] -= 1
                    selected_blocks.pop(i)
                    improved = True
                    swaps_done += 1
                    break
                
                candidates = set(blocks_covering_subset[critical_subsets[0]])
                candidates.discard(b_idx)
                
                for s in critical_subsets[1:]:
                    candidates.intersection_update(blocks_covering_subset[s])
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                    
                selected_set = set(selected_blocks)
                valid_candidates = [c for c in candidates if c not in selected_set]
                
                if not valid_candidates:
                    continue
                
                b_new = random.choice(valid_candidates)
                
                for s in block_subsets[b_idx]:
                    cover_count[s] -= 1
                for s in block_subsets[b_new]:
                    cover_count[s] += 1
                    
                selected_blocks[i] = b_new
                improved = True
                swaps_done += 1
                break
                
            if not improved:
                break
                
        # Final redundancy removal
        changed = True
        while changed:
            changed = False
            for i in range(len(selected_blocks) - 1, -1, -1):
                b_idx = selected_blocks[i]
                is_redundant = True
                for s in block_subsets[b_idx]:
                    if cover_count[s] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for s in block_subsets[b_idx]:
                        cover_count[s] -= 1
                    selected_blocks.pop(i)
                    changed = True
                    
        return selected_blocks, cover_count

    # 1. Run Greedy
    greedy_res, greedy_counts = run_greedy()
    
    if greedy_res is None:
        # Timeout in greedy? Return best effort or empty?
        # Should not happen.
        return []

    # 2. Local Search with multiple trials
    best_blocks = greedy_res
    best_len = len(greedy_res)
    
    # Determine how many trials we can afford
    t_elapsed = time.time() - start_time
    time_left = timeout - t_elapsed
    
    # Estimate time per local search trial.
    # It's hard to estimate exactly, but let's try a few.
    # If we have 100s left, we can do many.
    # Let's do up to 20 trials, or until time runs out.
    
    num_trials = 20
    for trial in range(num_trials):
        if time.time() - start_time > timeout - 5:
            break
            
        # Deep copy the greedy result
        trial_blocks, trial_counts = local_search_1swap(greedy_res, greedy_counts)
        
        if len(trial_blocks) < best_len:
            best_len = len(trial_blocks)
            best_blocks = trial_blocks
            
    final_blocks = [list(all_blocks[i]) for i in best_blocks]
    return final_blocks
