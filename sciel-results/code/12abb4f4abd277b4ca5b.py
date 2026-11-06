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
    
    # Precompute numpy arrays for fast operations
    block_subset_arrays = [np.array(subs, dtype=np.int32) for subs in block_to_subsets]
    subset_block_arrays = [np.array(blocks, dtype=np.int32) for blocks in subset_to_blocks]
    
    # Precompute subset_to_blocks as sets for fast intersection in swap
    subset_to_blocks_sets = [set(blocks) for blocks in subset_to_blocks]
    
    # Build a flat array of all (block, subset) pairs for vectorized gain updates
    # block_subset_flat: for each block, the list of subset indices it covers
    # We'll use this to vectorize gain updates
    
    # Precompute unique coverage for each block (how many subsets it uniquely covers given a coverage array)
    # This is computed on-the-fly in local search
    
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
                
            noise = rng.uniform(0, 0.3, size=num_blocks)
            scores = gains.astype(np.float64) + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_subset_arrays[best_idx]
            newly_covered = b_subs[uncovered[b_subs]]
            
            if len(newly_covered) == 0:
                continue
                
            uncovered[newly_covered] = False
            # Vectorized gain update using np.add.at
            for s in newly_covered:
                blocks_covering = subset_block_arrays[s]
                np.add.at(gains, blocks_covering, -1)
                    
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
            blocks_covering = subset_block_arrays[s_idx]
            np.add.at(gains, blocks_covering, 1)
                
        selected_blocks = []
        current_uncovered = uncovered_mask.copy()
        
        while True:
            if not np.any(current_uncovered):
                break
                
            noise = rng.uniform(0, 0.3, size=num_blocks)
            scores = gains.astype(np.float64) + noise
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
                blocks_covering = subset_block_arrays[s]
                np.add.at(gains, blocks_covering, -1)
                    
        return selected_blocks

    def minimize_with_swaps(blocks_indices, coverage_counts, time_limit, random_seed=None):
        """Local search with deletion and swap operations."""
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_blocks = list(blocks_indices)
        
        if not current_blocks:
            return current_blocks, coverage_counts.copy()
            
        # Work with a copy of coverage counts
        cover_count = coverage_counts.copy()
        
        # Use a set for O(1) membership testing
        current_blocks_set = set(current_blocks)
        
        improved = True
        start_time = time.time()
        
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            
            # Deletion pass: scan all blocks once, remove redundant ones without restart
            i = 0
            n = len(current_blocks)
            while i < n:
                if (time.time() - start_time) > time_limit:
                    break
                b_idx = current_blocks[i]
                b_subs = block_subset_arrays[b_idx]
                # Vectorized check: is any subset uniquely covered?
                is_redundant = not np.any(cover_count[b_subs] == 1)
                if is_redundant:
                    np.add.at(cover_count, b_subs, -1)
                    current_blocks_set.discard(b_idx)
                    current_blocks.pop(i)
                    n -= 1
                    improved = True
                    # Do NOT restart; continue scanning from same index
                else:
                    i += 1
            
            # Swap pass
            if (time.time() - start_time) > time_limit:
                break
                
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            
            swaps_done = 0
            max_swaps = 100
            
            for i in indices:
                if (time.time() - start_time) > time_limit:
                    break
                if swaps_done >= max_swaps:
                    break
                if i >= len(current_blocks):
                    continue
                    
                old_block = current_blocks[i]
                old_subs = block_subset_arrays[old_block]
                
                # Find unique subsets covered by old_block (vectorized)
                unique_mask = cover_count[old_subs] == 1
                unique_subs = old_subs[unique_mask]
                
                if len(unique_subs) == 0:
                    # Block is not critical - try swapping with a random block
                    for _attempt in range(5):
                        new_block = rng.randint(0, num_blocks - 1)
                        if new_block == old_block or new_block in current_blocks_set:
                            continue
                        break
                    else:
                        continue
                        
                    # Perform swap
                    np.add.at(cover_count, old_subs, -1)
                    new_subs = block_subset_arrays[new_block]
                    np.add.at(cover_count, new_subs, 1)
                    current_blocks_set.discard(old_block)
                    current_blocks_set.add(new_block)
                    current_blocks[i] = new_block
                    swaps_done += 1
                else:
                    # Block is critical - find candidates that cover all unique_subs
                    candidates = None
                    for s_idx in unique_subs:
                        if candidates is None:
                            candidates = subset_to_blocks_sets[s_idx].copy()
                        else:
                            candidates.intersection_update(subset_to_blocks_sets[s_idx])
                            if not candidates:
                                break
                    
                    if candidates:
                        candidates.discard(old_block)
                        if candidates:
                            # Pick a random candidate efficiently
                            cand_list = list(candidates)
                            new_block = cand_list[rng.randint(0, len(cand_list) - 1)]
                            if new_block in current_blocks_set:
                                continue
                                
                            # Perform swap
                            np.add.at(cover_count, old_subs, -1)
                            new_subs = block_subset_arrays[new_block]
                            np.add.at(cover_count, new_subs, 1)
                            current_blocks_set.discard(old_block)
                            current_blocks_set.add(new_block)
                            current_blocks[i] = new_block
                            swaps_done += 1
                            improved = True
        
        return current_blocks, cover_count

    def compute_coverage(blocks_indices):
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in blocks_indices:
            b_subs = block_subset_arrays[b_idx]
            cover_count[b_subs] += 1
        return cover_count

    def lll_random_construction(random_seed=None, p=None):
        """LLL-inspired probabilistic construction.
        
        Sample blocks with probability p, then iteratively remove locally redundant blocks.
        A block is locally redundant if none of its covered subsets are uniquely covered by it
        within the current set of selected blocks.
        """
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
        
        if p is None:
            from math import comb
            p_block = comb(k, t) / comb(v, t)
            p = 0.12
        
        # Sample blocks
        mask = rng.random(num_blocks) < p
        selected = np.where(mask)[0].tolist()
        
        if not selected:
            return []
        
        # Compute coverage
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in selected:
            b_subs = block_subset_arrays[b_idx]
            np.add.at(cover_count, b_subs, 1)
        
        # Check if all subsets are covered
        if np.any(cover_count == 0):
            uncovered_mask = (cover_count == 0).astype(bool)
            repair_res = run_greedy_partial(uncovered_mask, random_seed=random_seed)
            selected.extend(repair_res)
            cover_count = np.zeros(num_subsets, dtype=np.int32)
            for b_idx in selected:
                b_subs = block_subset_arrays[b_idx]
                np.add.at(cover_count, b_subs, 1)
        
        # Iteratively remove locally redundant blocks
        changed = True
        start_time = time.time()
        while changed and (time.time() - start_time) < 2.0:
            changed = False
            rng.shuffle(selected)
            i = 0
            n = len(selected)
            while i < n:
                if (time.time() - start_time) > 2.0:
                    break
                b_idx = selected[i]
                b_subs = block_subset_arrays[b_idx]
                is_redundant = not np.any(cover_count[b_subs] == 1)
                if is_redundant:
                    np.add.at(cover_count, b_subs, -1)
                    selected.pop(i)
                    n -= 1
                    changed = True
                    # Continue from same index
                else:
                    i += 1
        
        return selected

    best_blocks = None
    best_len = float('inf')
    best_coverage = None
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    trial = 0
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 8.0:
            break
            
        # Strategy 1: Full Greedy + Local Search (deletion + swap)
        greedy_res = run_greedy_full(trial)
        coverage = compute_coverage(greedy_res)
        
        # Adaptive time limit for local search
        ls_time = min(4.0, max(1.5, remaining * 0.12))
        min_res, min_cov = minimize_with_swaps(greedy_res, coverage, time_limit=ls_time, random_seed=trial)
        
        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res
            best_coverage = min_cov
            
        # Strategy 2: LLL-inspired probabilistic construction + Local Search
        if time.time() < deadline - 8:
            p_vals = [0.08, 0.10, 0.12, 0.15, 0.18]
            p = p_vals[trial % len(p_vals)]
            lll_res = lll_random_construction(random_seed=trial + 10000, p=p)
            
            if lll_res:
                lll_coverage = compute_coverage(lll_res)
                
                ls_time_lla = min(4.0, max(1.5, (deadline - time.time()) * 0.12))
                min_res_lla, min_cov_lla = minimize_with_swaps(lll_res, lll_coverage, time_limit=ls_time_lla, random_seed=trial + 20000)
                
                if len(min_res_lla) < best_len:
                    best_len = len(min_res_lla)
                    best_blocks = min_res_lla
                    best_coverage = min_cov_lla

        # Strategy 3: Perturb Best + Repair + Local Search
        if best_blocks is not None and len(best_blocks) > 20 and time.time() < deadline - 8:
            perturbed = list(best_blocks)
            num_remove = min(max(2, len(perturbed) // (12 + trial % 8)), len(perturbed) - 1)
            if num_remove > 0:
                cover_count_temp = best_coverage.copy()
                unique_counts = []
                for b_idx in perturbed:
                    b_subs = block_subset_arrays[b_idx]
                    unique = int(np.sum(cover_count_temp[b_subs] == 1))
                    unique_counts.append(unique)
                
                sorted_indices = sorted(range(len(perturbed)), key=lambda i: unique_counts[i])
                candidates = sorted_indices[:max(num_remove * 2, 4)]
                remove_indices = random.sample(candidates, min(num_remove, len(candidates)))
                blocks_to_remove = set(perturbed[i] for i in remove_indices)
                
                cover_count_temp2 = np.zeros(num_subsets, dtype=np.int32)
                for b_idx in perturbed:
                    if b_idx in blocks_to_remove:
                        continue
                    b_subs = block_subset_arrays[b_idx]
                    np.add.at(cover_count_temp2, b_subs, 1)
                        
                uncovered_mask = (cover_count_temp2 == 0).astype(bool)
                
                repair_res = run_greedy_partial(uncovered_mask, trial + 30000)
                
                new_set = [b for b in perturbed if b not in blocks_to_remove]
                new_set.extend(repair_res)
                
                new_coverage = compute_coverage(new_set)
                
                ls_time2 = min(4.0, max(1.5, (deadline - time.time()) * 0.12))
                min_res2, min_cov2 = minimize_with_swaps(new_set, new_coverage, time_limit=ls_time2, random_seed=trial + 40000)
                
                if len(min_res2) < best_len:
                    best_len = len(min_res2)
                    best_blocks = min_res2
                    best_coverage = min_cov2

        # Strategy 4: Pure swap-based diversification on best solution
        if best_blocks is not None and len(best_blocks) > 20 and time.time() < deadline - 8:
            ls_time3 = min(5.0, max(1.5, (deadline - time.time()) * 0.18))
            swap_res, swap_cov = minimize_with_swaps(best_blocks, best_coverage, time_limit=ls_time3, random_seed=trial + 50000)
            
            if len(swap_res) < best_len:
                best_len = len(swap_res)
                best_blocks = swap_res
                best_coverage = swap_cov

        trial += 1
        
    # Final polish: aggressive deletion pass on best solution
    if best_blocks is not None and time.time() < deadline - 5:
        final_cov = best_coverage.copy()
        final_blocks = list(best_blocks)
        final_set = set(final_blocks)
        for _ in range(8):
            changed = False
            i = 0
            n = len(final_blocks)
            while i < n and time.time() < deadline - 5:
                b_idx = final_blocks[i]
                b_subs = block_subset_arrays[b_idx]
                redundant = not np.any(final_cov[b_subs] == 1)
                if redundant:
                    np.add.at(final_cov, b_subs, -1)
                    final_set.discard(b_idx)
                    final_blocks.pop(i)
                    n -= 1
                    changed = True
                    # Continue from same index, don't restart
                else:
                    i += 1
            if not changed:
                break
        if len(final_blocks) < best_len:
            best_len = len(final_blocks)
            best_blocks = final_blocks
            best_coverage = final_cov
        
    if best_blocks is None:
        greedy_res = run_greedy_full(0)
        coverage = compute_coverage(greedy_res)
        best_blocks, best_coverage = minimize_with_swaps(greedy_res, coverage, time_limit=5.0, random_seed=0)
        
    return [list(all_blocks[i]) for i in best_blocks]
