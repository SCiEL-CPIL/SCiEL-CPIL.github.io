import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    points = range(v)
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute block_to_subsets and subset_to_blocks
    block_to_subsets = [[] for _ in range(num_blocks)]
    subset_to_blocks = [[] for _ in range(num_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        for s in itertools.combinations(block, t):
            s_idx = subset_to_idx[s]
            block_to_subsets[b_idx].append(s_idx)
            subset_to_blocks[s_idx].append(b_idx)
            
    # Convert to numpy arrays for faster operations where possible
    # block_subset_arrays is a list of arrays, which is fine for indexing
    block_subset_arrays = [np.array(subs, dtype=np.int32) for subs in block_to_subsets]
    subset_block_arrays = [np.array(blocks, dtype=np.int32) for blocks in subset_to_blocks]
    
    def run_greedy_full(random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered = np.ones(num_subsets, dtype=bool)
        gains = np.array([len(subs) for subs in block_to_subsets], dtype=np.int32)
        
        selected_blocks = []
        
        while True:
            if not np.any(uncovered):
                break
                
            # Add noise to break ties and explore
            noise = rng.uniform(0, 0.3, size=num_blocks)
            scores = gains + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_subset_arrays[best_idx]
            newly_covered = b_subs[uncovered[b_subs]]
            
            if len(newly_covered) == 0:
                continue
                
            uncovered[newly_covered] = False
            for s in newly_covered:
                for b in subset_block_arrays[s]:
                    gains[b] -= 1
                    
        return selected_blocks

    def run_greedy_partial(uncovered_mask, random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            return []
            
        gains = np.zeros(num_blocks, dtype=np.int32)
        for s_idx in uncovered_indices:
            for b_idx in subset_block_arrays[s_idx]:
                gains[b_idx] += 1
                
        selected_blocks = []
        current_uncovered = uncovered_mask.copy()
        
        while True:
            if not np.any(current_uncovered):
                break
                
            noise = rng.uniform(0, 0.3, size=num_blocks)
            scores = gains + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_subset_arrays[best_idx]
            newly_covered = b_subs[current_uncovered[b_subs]]
            
            if len(newly_covered) == 0:
                continue
                
            current_uncovered[newly_covered] = False
            for s in newly_covered:
                for b in subset_block_arrays[s]:
                    gains[b] -= 1
                    
        return selected_blocks

    def minimize_local_search(blocks_indices, random_seed=None):
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_blocks = set(blocks_indices)
        
        if not current_blocks:
            return list(current_blocks)
            
        # Initialize cover count
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            for s_idx in block_subset_arrays[b_idx]:
                cover_count[s_idx] += 1
                
        improved = True
        while improved:
            improved = False
            
            # 1. Remove redundant blocks
            redundant_found = False
            for b_idx in list(current_blocks):
                is_redundant = True
                for s_idx in block_subset_arrays[b_idx]:
                    if cover_count[s_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove it
                    current_blocks.remove(b_idx)
                    for s_idx in block_subset_arrays[b_idx]:
                        cover_count[s_idx] -= 1
                    improved = True
                    redundant_found = True
                    break # Restart scan after removal to handle cascading
            
            if redundant_found:
                continue
                
            # 2. Try swaps (1-1 exchange)
            # For each block, find if it can be replaced by another block 
            # that covers all its unique subsets.
            swap_performed = False
            # Shuffle blocks to avoid bias
            blocks_list = list(current_blocks)
            rng.shuffle(blocks_list)
            
            for b_out in blocks_list:
                # Find unique subsets for b_out
                unique_subsets = []
                for s_idx in block_subset_arrays[b_out]:
                    if cover_count[s_idx] == 1:
                        unique_subsets.append(s_idx)
                        
                if not unique_subsets:
                    continue # Should have been caught as redundant
                
                # Find a block b_in that covers all unique_subsets
                # b_in must be in the intersection of subset_to_blocks[s] for s in unique_subsets
                # And b_in must not be in current_blocks
                
                # Start with the first subset's blocks
                candidate_blocks = set(subset_block_arrays[unique_subsets[0]])
                
                # Intersect with others
                for s_idx in unique_subsets[1:]:
                    candidate_blocks.intersection_update(subset_block_arrays[s_idx])
                    if not candidate_blocks:
                        break
                
                # Remove current blocks from candidates
                candidate_blocks -= current_blocks
                
                if candidate_blocks:
                    # Pick a candidate (randomly or best)
                    # To keep it fast, pick one randomly
                    b_in = rng.choice(list(candidate_blocks))
                    
                    # Perform swap
                    current_blocks.remove(b_out)
                    current_blocks.add(b_in)
                    
                    # Update cover counts
                    for s_idx in block_subset_arrays[b_out]:
                        cover_count[s_idx] -= 1
                    for s_idx in block_subset_arrays[b_in]:
                        cover_count[s_idx] += 1
                    
                    improved = True
                    swap_performed = True
                    break # Restart loop after swap
                
        return list(current_blocks)

    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    trial = 0
    while time.time() < deadline:
        # Strategy 1: Full Greedy + Local Search Minimize
        greedy_res = run_greedy_full(trial)
        min_res = minimize_local_search(greedy_res, trial)
        
        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res
            
        # Strategy 2: Perturb Best + Repair + Minimize
        if best_blocks is not None and len(best_blocks) > 20 and time.time() < deadline - 8:
            perturbed = list(best_blocks)
            num_remove = min(max(2, len(perturbed) // 20), len(perturbed) - 1)
            if num_remove > 0:
                remove_indices = random.sample(range(len(perturbed)), num_remove)
                blocks_to_remove = set(perturbed[i] for i in remove_indices)
                
                cover_count_temp = np.zeros(num_subsets, dtype=np.int32)
                for b_idx in perturbed:
                    if b_idx in blocks_to_remove:
                        continue
                    for s_idx in block_subset_arrays[b_idx]:
                        cover_count_temp[s_idx] += 1
                        
                uncovered_mask = (cover_count_temp == 0).astype(bool)
                
                repair_res = run_greedy_partial(uncovered_mask, trial + 1000)
                
                new_set = [b for b in perturbed if b not in blocks_to_remove]
                new_set.extend(repair_res)
                
                min_res2 = minimize_local_search(new_set, trial + 2000)
                
                if len(min_res2) < best_len:
                    best_len = len(min_res2)
                    best_blocks = min_res2

        trial += 1
        
    if best_blocks is None:
        greedy_res = run_greedy_full(0)
        best_blocks = minimize_local_search(greedy_res, 0)
        
    return [list(all_blocks[i]) for i in best_blocks]
