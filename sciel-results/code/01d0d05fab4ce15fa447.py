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
            
    # Convert to numpy arrays for faster access
    block_to_subsets_arr = [np.array(s, dtype=np.int32) for s in block_to_subsets]
    subset_to_blocks_arr = [np.array(s, dtype=np.int32) for s in subset_to_blocks]
    
    # Helper: Greedy Set Cover
    def run_greedy(cover_count, random_seed=None, noise_scale=0.0):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered_mask = (cover_count == 0).astype(bool)
        if not np.any(uncovered_mask):
            return []
            
        # Initial gains: for each block, how many uncovered subsets it covers
        # We can compute this by iterating uncovered subsets and incrementing gains for blocks covering them
        gains = np.zeros(num_blocks, dtype=np.int32)
        uncovered_indices = np.where(uncovered_mask)[0]
        for s_idx in uncovered_indices:
            blocks_covering = subset_to_blocks_arr[s_idx]
            gains[blocks_covering] += 1
            
        selected_blocks = []
        uncovered_count = np.sum(uncovered_mask)
        
        while uncovered_count > 0:
            if noise_scale > 0:
                noise = rng.uniform(0, noise_scale, size=num_blocks)
                scores = gains + noise
            else:
                scores = gains
                
            best_idx = np.argmax(scores)
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_to_subsets_arr[best_idx]
            newly_covered = b_subs[uncovered_mask[b_subs]]
            
            if len(newly_covered) == 0:
                continue
                
            cover_count[newly_covered] += 1
            uncovered_mask[newly_covered] = False
            
            # Update gains
            for s in newly_covered:
                blocks_covering_s = subset_to_blocks_arr[s]
                gains[blocks_covering_s] -= 1
                
            uncovered_count -= len(newly_covered)
            
        return selected_blocks

    def minimize(blocks_indices, random_seed=None):
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_blocks = list(blocks_indices)
        if not current_blocks:
            return current_blocks
            
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            cover_count[block_to_subsets_arr[b_idx]] += 1
            
        changed = True
        while changed:
            changed = False
            redundant_blocks = []
            
            for i, b_idx in enumerate(current_blocks):
                b_subs = block_to_subsets_arr[b_idx]
                if not np.any(cover_count[b_subs] == 1):
                    redundant_blocks.append(i)
                    
            if redundant_blocks:
                remove_pos = redundant_blocks[rng.randint(0, len(redundant_blocks)-1)]
                b_idx_to_remove = current_blocks[remove_pos]
                cover_count[block_to_subsets_arr[b_idx_to_remove]] -= 1
                current_blocks.pop(remove_pos)
                changed = True
                
        return current_blocks

    def local_search_swap(blocks_indices, max_swaps, random_seed):
        """
        Perform local search using swaps.
        Returns a new list of blocks if improved, else None.
        """
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_blocks = list(blocks_indices)
        if not current_blocks:
            return current_blocks
            
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            cover_count[block_to_subsets_arr[b_idx]] += 1
            
        block_set = set(current_blocks)
        
        # Try to find swaps
        # We iterate through blocks and try to replace them
        # To avoid O(N^2), we sample blocks to try
        num_tries = min(len(current_blocks) * 2, 50)
        
        improved = False
        for _ in range(num_tries):
            if not current_blocks:
                break
                
            # Pick a random block to potentially swap
            idx = rng.randint(0, len(current_blocks) - 1)
            b_idx = current_blocks[idx]
            b_subs = block_to_subsets_arr[b_idx]
            
            # Find subsets in b_idx that are covered ONLY by b_idx
            critical_subsets = b_subs[cover_count[b_subs] == 1]
            
            if len(critical_subsets) == 0:
                # This block is redundant!
                # Remove it
                cover_count[b_subs] -= 1
                current_blocks.pop(idx)
                block_set.remove(b_idx)
                improved = True
                continue
            
            # Find a replacement block B that covers all critical_subsets
            # Candidate blocks are the intersection of subset_to_blocks[s] for s in critical_subsets
            # Start with the set of blocks covering the first critical subset
            candidate_blocks = set(subset_to_blocks_arr[critical_subsets[0]].tolist())
            
            for s in critical_subsets[1:]:
                candidate_blocks.intersection_update(subset_to_blocks_arr[s].tolist())
                if not candidate_blocks:
                    break
            
            if not candidate_blocks:
                continue
                
            # Remove current blocks from candidates
            candidate_blocks.difference_update(block_set)
            
            if not candidate_blocks:
                continue
                
            # Pick a candidate
            # Prefer one that covers many other subsets (high "weight") to help future deletions
            # Or just random
            candidates_list = list(candidate_blocks)
            new_b_idx = candidates_list[rng.randint(0, len(candidates_list) - 1)]
            
            # Perform swap
            # Remove old
            cover_count[block_to_subsets_arr[b_idx]] -= 1
            current_blocks[idx] = new_b_idx
            block_set.remove(b_idx)
            block_set.add(new_b_idx)
            # Add new
            cover_count[block_to_subsets_arr[new_b_idx]] += 1
            improved = True
            
        return current_blocks

    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    trial = 0
    
    # Phase 1: Initial Greedy
    cover_count_zero = np.zeros(num_subsets, dtype=np.int32)
    greedy_res = run_greedy(cover_count_zero, random_seed=0, noise_scale=0.0)
    min_res = minimize(greedy_res, random_seed=0)
    best_blocks = min_res
    best_len = len(min_res)
    
    # Phase 2: Iterative Improvement
    while time.time() < deadline:
        # Perturb: Remove some blocks
        num_remove = rng.randint(1, 5) if 'rng' in locals() else random.randint(1, 5)
        if len(best_blocks) <= num_remove:
            num_remove = max(1, len(best_blocks) // 2)
            
        remove_indices = random.sample(range(len(best_blocks)), num_remove)
        blocks_to_remove_set = set(best_blocks[i] for i in remove_indices)
        remaining_blocks = [b for b in best_blocks if b not in blocks_to_remove_set]
        
        # Calculate cover count for remaining
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in remaining_blocks:
            cover_count[block_to_subsets_arr[b_idx]] += 1
            
        # Repair
        repair_res = run_greedy(cover_count, random_seed=trial + 1000, noise_scale=0.1)
        
        # Combine
        new_blocks = remaining_blocks + repair_res
        
        # Minimize
        min_res = minimize(new_blocks, random_seed=trial + 1001)
        
        # Local Search (Swaps)
        # Only if we have time
        if time.time() < deadline - 5.0:
            ls_res = local_search_swap(min_res, max_swaps=10, random_seed=trial + 1002)
            if ls_res is not None:
                # Minimize again after swaps
                min_res = minimize(ls_res, random_seed=trial + 1003)
                
        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res
            
        trial += 1

    return [list(all_blocks[i]) for i in best_blocks]
