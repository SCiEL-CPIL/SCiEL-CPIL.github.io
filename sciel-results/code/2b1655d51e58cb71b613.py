import itertools
import random
import time
import sys
import numpy as np

def construct(v, k, t):
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # Map each block to the set of t-subset indices it covers
    block_t_indices = [set() for _ in range(num_k_subsets)]
    for b_idx, block in enumerate(all_k_subsets):
        for s in itertools.combinations(block, t):
            block_t_indices[b_idx].add(t_subset_to_idx[s])
            
    # Map each t-subset to the blocks that cover it
    t_subset_blocks = [set() for _ in range(num_t_subsets)]
    for b_idx in range(num_k_subsets):
        for t_idx in block_t_indices[b_idx]:
            t_subset_blocks[t_idx].add(b_idx)
    
    # Build coverage matrix: rows = blocks, cols = t-subsets, 1 if block covers t-subset
    # Use numpy for fast operations
    cover_matrix = np.zeros((num_k_subsets, num_t_subsets), dtype=np.int8)
    for b_idx in range(num_k_subsets):
        for t_idx in block_t_indices[b_idx]:
            cover_matrix[b_idx, t_idx] = 1
            
    # 2. Phase 1: Greedy Independent Set (Core)
    pool = set(range(num_k_subsets))
    core_blocks = set()
    covered_by_core = set()
    
    random.seed(42)
    
    while pool:
        b_idx = random.choice(list(pool))
        core_blocks.add(b_idx)
        
        # Find neighbors to remove
        to_remove = set()
        for t_idx in block_t_indices[b_idx]:
            to_remove.update(t_subset_blocks[t_idx])
            
        pool.difference_update(to_remove)
        
        for t_idx in block_t_indices[b_idx]:
            covered_by_core.add(t_idx)

    # 3. Phase 2: Greedy Set Cover for Residual
    residual_uncovered = set(range(num_t_subsets)) - covered_by_core
    
    if not residual_uncovered:
        result = [list(all_k_subsets[b]) for b in core_blocks]
        return result
    
    # Use numpy for faster greedy
    # residual_uncovered as boolean array
    uncovered_arr = np.zeros(num_t_subsets, dtype=bool)
    for t_idx in residual_uncovered:
        uncovered_arr[t_idx] = True
        
    selected_blocks = set(core_blocks)
    
    # Precompute: for each block, list of t-subsets it covers (as array)
    block_t_arrs = []
    for b_idx in range(num_k_subsets):
        block_t_arrs.append(np.array(list(block_t_indices[b_idx]), dtype=np.int32))
    
    # Greedy: repeatedly pick block with max coverage of uncovered
    # Use numpy to compute gains quickly
    # But iterating all blocks each time is still O(N*M) where M is #t-subsets
    # Better: maintain for each uncovered t-subset, which blocks cover it
    # Then for each block, gain = number of uncovered t-subsets it covers
    
    # Alternative: use the cover_matrix
    # gain for block b = sum(cover_matrix[b] & uncovered_arr)
    
    max_iterations = 500  # Safety
    for _ in range(max_iterations):
        if not uncovered_arr.any():
            break
            
        # Compute gains for all blocks not yet selected
        gains = (cover_matrix & uncovered_arr[np.newaxis, :]).sum(axis=1)
        
        # Mask out already selected blocks
        for b_idx in selected_blocks:
            gains[b_idx] = -1
            
        best_b = int(np.argmax(gains))
        if gains[best_b] <= 0:
            break
            
        selected_blocks.add(best_b)
        uncovered_arr[cover_matrix[best_b] == 1] = False
        
    # 4. Local Search / Minimalization
    
    active_blocks = set(selected_blocks)
    cover_count = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in active_blocks:
        for t_idx in block_t_indices[b_idx]:
            cover_count[t_idx] += 1
            
    def minimize(active_blocks_set, cover_counts):
        changed = True
        while changed:
            changed = False
            items = list(active_blocks_set)
            random.shuffle(items)
            for b_idx in items:
                if b_idx not in active_blocks_set:
                    continue
                can_remove = True
                for t_idx in block_t_indices[b_idx]:
                    if cover_counts[t_idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    active_blocks_set.remove(b_idx)
                    for t_idx in block_t_indices[b_idx]:
                        cover_counts[t_idx] -= 1
                    changed = True
        return active_blocks_set, cover_counts

    active_blocks, cover_count = minimize(active_blocks, cover_count)
    
    # Swap search with time limit
    start_time = time.time()
    time_limit = 110.0
    deadline = start_time + time_limit
    
    while time.time() < deadline:
        improved = False
        current_blocks = list(active_blocks)
        random.shuffle(current_blocks)
        
        for B in current_blocks:
            if B not in active_blocks:
                continue
            if time.time() > deadline:
                break
            
            # Find t-subsets uniquely covered by B
            uncovered_by_B = []
            for t_idx in block_t_indices[B]:
                if cover_count[t_idx] == 1:
                    uncovered_by_B.append(t_idx)
            
            if not uncovered_by_B:
                continue

            # Sort by number of covering blocks
            uncovered_by_B.sort(key=lambda x: len(t_subset_blocks[x]))
            
            candidates = None
            for t_idx in uncovered_by_B:
                blocks_list = t_subset_blocks[t_idx]
                if candidates is None:
                    candidates = set(blocks_list)
                else:
                    candidates.intersection_update(blocks_list)
                
                if not candidates:
                    break
            
            if not candidates:
                continue
            
            # Filter candidates: must cover all uniquely covered by B, and not be B
            valid_candidates = []
            for c in candidates:
                if c != B and c not in active_blocks:
                    valid_candidates.append(c)
                    
            if not valid_candidates:
                continue
            
            B_new = random.choice(valid_candidates)
            
            # Check if swap improves or maintains coverage
            # Remove B, add B_new, then minimize
            active_blocks.remove(B)
            active_blocks.add(B_new)
            
            for t_idx in block_t_indices[B]:
                cover_count[t_idx] -= 1
            for t_idx in block_t_indices[B_new]:
                cover_count[t_idx] += 1
            
            # Check validity
            valid = True
            for t_idx in block_t_indices[B]:
                if cover_count[t_idx] == 0:
                    valid = False
                    break
            
            if not valid:
                # Revert
                active_blocks.remove(B_new)
                active_blocks.add(B)
                for t_idx in block_t_indices[B]:
                    cover_count[t_idx] += 1
                for t_idx in block_t_indices[B_new]:
                    cover_count[t_idx] -= 1
                continue
            
            # Minimize after swap
            active_blocks, cover_count = minimize(active_blocks, cover_count)
            improved = True
            
        if not improved:
            # Try a few random restarts of local search
            for _ in range(3):
                if time.time() > deadline:
                    break
                # Random perturbation: remove a random block, then re-add greedily
                if len(active_blocks) > 10:
                    B_rand = random.choice(list(active_blocks))
                    active_blocks.remove(B_rand)
                    for t_idx in block_t_indices[B_rand]:
                        cover_count[t_idx] -= 1
                    
                    # Re-cover any uncovered t-subsets
                    uncovered_now = [t_idx for t_idx in range(num_t_subsets) if cover_count[t_idx] == 0]
                    for t_idx in uncovered_now:
                        # Find a block covering this t-subset
                        covering = [b for b in t_subset_blocks[t_idx] if b not in active_blocks]
                        if covering:
                            best_b = random.choice(covering)
                            active_blocks.add(best_b)
                            for t_idx2 in block_t_indices[best_b]:
                                cover_count[t_idx2] += 1
                    
                    active_blocks, cover_count = minimize(active_blocks, cover_count)

    result = [list(all_k_subsets[b_idx]) for b_idx in active_blocks]
    
    # Final validation
    final_cover = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in active_blocks:
        for t_idx in block_t_indices[b_idx]:
            final_cover[t_idx] += 1
    
    if (final_cover > 0).all():
        return result
    
    # Fallback: return greedy solution
    return [list(all_k_subsets[b_idx]) for b_idx in selected_blocks]
