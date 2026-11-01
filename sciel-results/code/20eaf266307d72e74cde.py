import itertools
import random
import time

def construct(v, k, t):
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # Map each block to the indices of the t-subsets it covers
    block_t_indices = []
    for block in all_k_subsets:
        indices = [t_subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_t_indices.append(indices)
        
    # Map each t-subset to the blocks that cover it
    t_subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_blocks[idx].append(b_idx)
            
    # Precompute degrees of t-subsets (number of blocks covering each)
    t_subset_degrees = [len(lst) for lst in t_subset_blocks]

    # 2. Phase 1: Weighted Greedy Independent Set (Packing)
    # We want to select blocks that do not share any t-subset.
    # Weight: Prefer blocks that cover "rare" t-subsets (low degree) to ensure hard parts are covered early.
    
    pool = set(range(num_k_subsets))
    core_blocks = []
    uncovered_mask = [1] * num_t_subsets
    
    # Calculate initial "scarcity" weight for each block
    # weight[b] = sum of 1/degree[t] for t in block
    # This is expensive to compute for all 50k blocks initially? 
    # 50k * C(7,4) = 50k * 35 = 1.75M operations. Acceptable.
    
    block_weights = [0.0] * num_k_subsets
    for b_idx, indices in enumerate(block_t_indices):
        w = 0.0
        for t_idx in indices:
            deg = t_subset_degrees[t_idx]
            if deg > 0:
                w += 1.0 / deg
        block_weights[b_idx] = w

    while pool:
        best_block = -1
        max_score = -1.0
        
        # To speed up, we might not scan the whole pool every time if it's huge, 
        # but 50k is small enough.
        for b_idx in pool:
            # Score = current uncovered coverage * scarcity factor
            # If a block covers many uncovered items, it's good.
            # If it covers rare items, it's good.
            
            # Quick check: if it covers 0 uncovered, skip?
            # We need to calculate the score.
            cov_uncovered = 0
            scarcity_sum = 0.0
            for t_idx in block_t_indices[b_idx]:
                if uncovered_mask[t_idx]:
                    cov_uncovered += 1
                    scarcity_sum += 1.0 / t_subset_degrees[t_idx]
            
            if cov_uncovered == 0:
                continue
                
            # Combine: Maximize covered count, tie-break by scarcity
            # Or: Maximize (cov_uncovered * 100 + scarcity_sum)
            score = cov_uncovered * 100.0 + scarcity_sum
            
            if score > max_score:
                max_score = score
                best_block = b_idx
                
        if best_block == -1:
            break
            
        core_blocks.append(best_block)
        
        # Update uncovered mask
        for t_idx in block_t_indices[best_block]:
            uncovered_mask[t_idx] = 0
            
        # Remove best_block and its conflicts from pool
        neighbors_to_remove = set()
        for t_idx in block_t_indices[best_block]:
            neighbors_to_remove.update(t_subset_blocks[t_idx])
        
        pool.difference_update(neighbors_to_remove)

    # 3. Phase 2: Solve Remaining Set Cover
    active_blocks = set(core_blocks)
    cover_count = [0] * num_t_subsets
    
    for b_idx in active_blocks:
        for t_idx in block_t_indices[b_idx]:
            cover_count[t_idx] += 1
            
    still_uncovered = [i for i in range(num_t_subsets) if cover_count[i] == 0]
    
    if still_uncovered:
        residual_uncovered = set(still_uncovered)
        
        while residual_uncovered:
            best_block = -1
            max_gain = 0
            
            # Standard Greedy: Pick block covering most residual uncovered
            # We can restrict to blocks not in active_blocks to avoid duplicates
            # But checking `not in active_blocks` for 50k blocks is O(N).
            # We can just iterate all and check gain. If block is in active, gain is 0 for new items?
            # No, if block is in active, it already covers its items. 
            # So we only consider blocks NOT in active_blocks.
            
            for b_idx in range(num_k_subsets):
                if b_idx in active_blocks:
                    continue
                gain = 0
                for t_idx in block_t_indices[b_idx]:
                    if t_idx in residual_uncovered:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block = b_idx
                    
            if best_block == -1 or max_gain == 0:
                break
                
            active_blocks.add(best_block)
            for t_idx in block_t_indices[best_block]:
                if t_idx in residual_uncovered:
                    residual_uncovered.remove(t_idx)
                cover_count[t_idx] += 1

    # 4. Minimize
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

    # 5. Local Search (Enhanced)
    start_time = time.time()
    time_limit = 110.0
    
    # Helper to try a move
    def try_remove(b_idx):
        nonlocal active_blocks, cover_count
        if b_idx not in active_blocks:
            return False
        # Check if all t-subsets covered by b_idx have count > 1
        for t_idx in block_t_indices[b_idx]:
            if cover_count[t_idx] <= 1:
                return False
        # Remove
        active_blocks.remove(b_idx)
        for t_idx in block_t_indices[b_idx]:
            cover_count[t_idx] -= 1
        return True

    def try_add(b_idx):
        nonlocal active_blocks, cover_count
        if b_idx in active_blocks:
            return False
        # Add
        active_blocks.add(b_idx)
        for t_idx in block_t_indices[b_idx]:
            cover_count[t_idx] += 1
        return True

    def try_swap(old_idx, new_idx):
        nonlocal active_blocks, cover_count
        if old_idx not in active_blocks or new_idx in active_blocks:
            return False
        # Check validity: new_idx must cover all unique subsets of old_idx?
        # No, we just swap and then minimize. But we should ensure we don't break coverage.
        # A safe swap: new_idx covers all t-subsets that are *uniquely* covered by old_idx.
        
        # Find unique subsets of old_idx
        unique_subsets = []
        for t_idx in block_t_indices[old_idx]:
            if cover_count[t_idx] == 1:
                unique_subsets.append(t_idx)
                
        if not unique_subsets:
            # Safe to swap with anything? 
            # If no unique subsets, removing old_idx leaves all its subsets covered by others.
            # So we can swap with any new_idx.
            pass
        else:
            # Check if new_idx covers all unique_subsets
            for t_idx in unique_subsets:
                if t_idx not in block_t_indices[new_idx]:
                    return False

        # Perform Swap
        active_blocks.remove(old_idx)
        active_blocks.add(new_idx)
        
        for t_idx in block_t_indices[old_idx]:
            cover_count[t_idx] -= 1
        for t_idx in block_t_indices[new_idx]:
            cover_count[t_idx] += 1
        return True

    while time.time() - start_time < time_limit:
        improved = False
        current_blocks = list(active_blocks)
        random.shuffle(current_blocks)
        
        for B in current_blocks:
            if B not in active_blocks:
                continue
            
            # Try Remove
            if try_remove(B):
                # Minimize after remove? It's already minimal regarding B.
                # But other blocks might become redundant.
                active_blocks, cover_count = minimize(active_blocks, cover_count)
                improved = True
                continue # B is gone, don't try other moves for B in this iteration?
                # Actually, we re-shuffled, so B is removed from list.
                # We can just continue to next B.
            
            # Try Swap
            # Find candidates for swap
            # Candidates must cover unique subsets of B
            unique_subsets = []
            for t_idx in block_t_indices[B]:
                if cover_count[t_idx] == 1:
                    unique_subsets.append(t_idx)
            
            if unique_subsets:
                # Intersect candidate lists
                candidates = None
                for t_idx in unique_subsets:
                    blocks_list = t_subset_blocks[t_idx]
                    if candidates is None:
                        candidates = set(blocks_list)
                    else:
                        candidates.intersection_update(blocks_list)
                    if not candidates:
                        break
                
                if candidates:
                    # Filter out B and active blocks
                    valid_candidates = [c for c in candidates if c != B and c not in active_blocks]
                    if valid_candidates:
                        # Pick one randomly
                        B_new = random.choice(valid_candidates)
                        if try_swap(B, B_new):
                            active_blocks, cover_count = minimize(active_blocks, cover_count)
                            improved = True
            else:
                # No unique subsets, B is "redundant" in terms of unique coverage.
                # We can swap with ANY block not in active? 
                # Or just try to remove? (Already tried remove, failed, so it's not redundant? 
                # Wait, if unique_subsets is empty, remove should have succeeded?
                # try_remove checks if cover_count[t] <= 1. If unique is empty, all cover_count > 1.
                # So try_remove should have succeeded.
                # So this case (unique empty but remove failed) shouldn't happen.
                pass
        
        # Try Add
        # Find a block that covers many uncovered? 
        # But we should have no uncovered. 
        # Maybe add a block and remove two? (2-for-1 swap)
        # This is complex. Let's stick to Add if we detect uncovered?
        # We assume cover is valid.
        
        if not improved:
            # Try a few random adds/removes to escape local minima?
            # Or break?
            # Let's try a random "perturbation": Add a random block, then minimize.
            # If size decreases or stays same, keep it.
            # Actually, minimize will remove redundant blocks.
            # If we add a block, size increases by 1, then minimize might remove 2 others?
            # This is a "2-move" local search.
            
            # Pick a random block not in active
            # We need a list of inactive blocks.
            # This is O(N). 
            # Let's just break if no improvement to save time, 
            # or do a few random swaps.
            break

    result = [list(all_k_subsets[b_idx]) for b_idx in active_blocks]
    return result
