import numpy as np
import itertools
import time
import random
import sys

def construct(v, k, t):
    points = range(v)
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute bitmasks for each block
    block_masks = [0] * num_blocks
    subset_to_blocks = [[] for _ in range(num_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for s in itertools.combinations(block, t):
            s_idx = subset_to_idx[s]
            mask |= (1 << s_idx)
            subset_to_blocks[s_idx].append(b_idx)
        block_masks[b_idx] = mask
        
    total_subsets_mask = (1 << num_subsets) - 1
    
    def run_greedy_full(random_seed=None):
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        covered = 0
        selected_blocks = []
        
        while covered != total_subsets_mask:
            best_gain = 0
            best_idx = -1
            
            for i in range(num_blocks):
                new_mask = block_masks[i] & ~covered
                if new_mask:
                    g = new_mask.bit_count()
                    if g > best_gain:
                        best_gain = g
                        best_idx = i
                    elif g == best_gain and g > 0:
                        if rng.random() < 0.5:
                            best_idx = i
            
            if best_idx == -1:
                break
            
            selected_blocks.append(best_idx)
            covered |= block_masks[best_idx]
            
        return selected_blocks

    def minimize(blocks_indices, random_seed=None, time_limit=5.0):
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_blocks = list(blocks_indices)
        if not current_blocks:
            return current_blocks
            
        covered = 0
        for b_idx in current_blocks:
            covered |= block_masks[b_idx]
            
        # Phase 1: Remove redundant blocks
        changed = True
        while changed:
            changed = False
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            for i in indices:
                b_idx = current_blocks[i]
                c_without_b = covered & ~block_masks[b_idx]
                unique = block_masks[b_idx] & ~c_without_b
                if unique == 0:
                    covered = c_without_b
                    current_blocks.pop(i)
                    changed = True
        
        # Phase 2: Swap-based local search
        current_set = set(current_blocks)
        improved = True
        max_iters = 100
        iter_count = 0
        start_min = time.time()
        
        while improved and iter_count < max_iters and time.time() - start_min < time_limit:
            improved = False
            iter_count += 1
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            
            for i in indices:
                b_idx = current_blocks[i]
                c_without_b = covered & ~block_masks[b_idx]
                unique = block_masks[b_idx] & ~c_without_b
                
                if unique == 0:
                    covered = c_without_b
                    current_blocks.pop(i)
                    current_set.remove(b_idx)
                    improved = True
                    continue
                
                lsb = unique & -unique
                s_idx = lsb.bit_length() - 1
                
                candidates = subset_to_blocks[s_idx]
                best_new = -1
                
                # Shuffle candidates to diversify?
                # candidates is a list. We can pick a random candidate to check first?
                # Or just iterate.
                # To speed up, we can check a few random candidates first.
                # But subset_to_blocks[s_idx] is not huge.
                
                for cand in candidates:
                    if cand in current_set or cand == b_idx:
                        continue
                    if (unique & ~block_masks[cand]) == 0:
                        best_new = cand
                        break
                
                if best_new != -1:
                    covered = (covered & ~block_masks[b_idx]) | block_masks[best_new]
                    current_blocks[i] = best_new
                    current_set.remove(b_idx)
                    current_set.add(best_new)
                    improved = True
        
        return current_blocks

    def repair(uncovered_mask, random_seed=None):
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        selected = []
        remaining = uncovered_mask
        
        while remaining:
            best_gain = 0
            best_idx = -1
            for i in range(num_blocks):
                new_mask = block_masks[i] & remaining
                if new_mask:
                    g = new_mask.bit_count()
                    if g > best_gain:
                        best_gain = g
                        best_idx = i
                    elif g == best_gain and g > 0:
                        if rng.random() < 0.5:
                            best_idx = i
            if best_idx == -1:
                break
            selected.append(best_idx)
            remaining &= ~block_masks[best_idx]
        return selected

    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    trial = 0
    while time.time() < deadline - 5.0:
        greedy_res = run_greedy_full(trial)
        min_res = minimize(greedy_res, trial, time_limit=5.0)
        
        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res
            
        if best_blocks is not None and len(best_blocks) > 10 and time.time() < deadline - 15.0:
            perturbed = list(best_blocks)
            num_remove = min(5, len(perturbed) - 1)
            if num_remove > 0:
                remove_indices = random.sample(range(len(perturbed)), num_remove)
                blocks_to_remove = set(perturbed[i] for i in remove_indices)
                
                covered_temp = 0
                for b_idx in perturbed:
                    if b_idx not in blocks_to_remove:
                        covered_temp |= block_masks[b_idx]
                
                uncovered_mask = total_subsets_mask & ~covered_temp
                
                repair_res = repair(uncovered_mask, trial + 1000)
                
                new_set = [b for b in perturbed if b not in blocks_to_remove]
                new_set.extend(repair_res)
                
                min_res2 = minimize(new_set, trial + 2000, time_limit=5.0)
                
                if len(min_res2) < best_len:
                    best_len = len(min_res2)
                    best_blocks = min_res2

        trial += 1
        
    if best_blocks is None:
        greedy_res = run_greedy_full(0)
        best_blocks = minimize(greedy_res, 0, time_limit=5.0)
        
    return [list(all_blocks[i]) for i in best_blocks]
