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
        
        # If there are unique subsets, new_idx MUST cover them
        if unique_subsets:
            # Check if new_idx covers all unique subsets
            new_set = set(block_t_indices[new_idx])
            for t_idx in unique_subsets:
                if t_idx not in new_set:
                    return False
        
        # Perform swap
        active_blocks.remove(old_idx)
        for t_idx in block_t_indices[old_idx]:
            cover_count[t_idx] -= 1
            
        active_blocks.add(new_idx)
        for t_idx in block_t_indices[new_idx]:
            cover_count[t_idx] += 1
        return True

    # Main Local Search Loop
    best_size = len(active_blocks)
    best_solution = list(active_blocks)
    rng = random.Random(42)
    
    while time.time() - start_time < time_limit:
        # Try random removals
        removed_any = False
        candidates = list(active_blocks)
        rng.shuffle(candidates)
        for b_idx in candidates:
            if try_remove(b_idx):
                removed_any = True
                # Try to minimize further after removal
                active_blocks, cover_count = minimize(active_blocks, cover_count)
                if len(active_blocks) < best_size:
                    best_size = len(active_blocks)
                    best_solution = list(active_blocks)
        
        # Try random swaps
        # Sample a few swaps
        for _ in range(50):
            if time.time() - start_time > time_limit:
                break
            old_idx = rng.choice(list(active_blocks))
            # Pick a random new_idx from all blocks not in active
            # To be efficient, we might just pick a random index and check
            new_idx = rng.randint(0, num_k_subsets - 1)
            if new_idx not in active_blocks:
                if try_swap(old_idx, new_idx):
                    # Minimize after swap
                    active_blocks, cover_count = minimize(active_blocks, cover_count)
                    if len(active_blocks) < best_size:
                        best_size = len(active_blocks)
                        best_solution = list(active_blocks)
                    else:
                        # If it didn't improve, revert? 
                        # Actually, try_swap modifies state. We need to revert if not improved.
                        # But minimizing might have removed others. 
                        # Simplest: keep state, but track best.
                        # Since we want to explore, we can keep the state even if size is same or larger?
                        # No, we want to minimize size. If size increased, we should probably revert.
                        # However, minimizing ensures we are at a local minimum for the current set.
                        # If size > best_size, it's bad. Let's just accept and let next iterations fix?
                        # Or revert. Reverting is safer for "best found".
                        # But we lost the old state.
                        # Let's just keep the current state. The "best_solution" is tracked separately.
                        pass

    return [list(all_k_subsets[b]) for b in best_solution]
