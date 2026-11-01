import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    # Precompute t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    
    # Precompute k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    # Build block_bitmasks: for each block, a bitmask of which t-subsets it covers
    block_bitmasks = [0] * n_k_blocks
    block_points = all_k_blocks
    
    for i, block in enumerate(all_k_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            mask |= (1 << idx)
        block_bitmasks[i] = mask
    
    # For each t-subset, list of blocks that cover it (for swap search)
    tsub_blocks = [[] for _ in range(n_t_subsets)]
    for bi in range(n_k_blocks):
        mask = block_bitmasks[bi]
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            tsub_blocks[idx].append(bi)
            m ^= lsb
    
    def greedy_cover():
        """Global greedy set cover using bitmask representation."""
        uncovered_mask = (1 << n_t_subsets) - 1
        selected_blocks = []
        
        while uncovered_mask:
            if time.time() > deadline:
                break
                
            best_idx = -1
            best_gain = 0
            
            for i in range(n_k_blocks):
                gain = (uncovered_mask & block_bitmasks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx == -1:
                break
                
            selected_blocks.append(best_idx)
            uncovered_mask &= ~block_bitmasks[best_idx]
            
        return selected_blocks
    
    def local_search(blocks_indices, time_limit=None):
        """Local search: delete redundant blocks and try swaps."""
        if time_limit is None:
            time_limit = deadline - time.time()
        start_ls = time.time()
        
        blocks_indices = list(blocks_indices)
        
        # Build coverage counts using NumPy
        cov = np.zeros(n_t_subsets, dtype=np.int32)
        for bi in blocks_indices:
            mask = block_bitmasks[bi]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cov[idx] += 1
                m ^= lsb
        
        while time.time() - start_ls < time_limit and time.time() < deadline:
            improved = False
            
            # Phase 1: Delete redundant blocks
            indices = list(range(len(blocks_indices)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                bi = blocks_indices[i]
                mask = block_bitmasks[bi]
                m = mask
                essential = False
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cov[idx] == 1:
                        essential = True
                        break
                    m ^= lsb
                if not essential:
                    # Remove this block
                    blocks_indices.pop(i)
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cov[idx] -= 1
                        m ^= lsb
                    improved = True
                    break
            
            if improved:
                continue
            
            # Phase 2: Try single-block swaps
            if len(blocks_indices) == 0:
                break
            
            found_swap = False
            num_tries = min(100, len(blocks_indices))
            
            for _ in range(num_tries):
                if time.time() > deadline:
                    break
                
                i = random.randrange(len(blocks_indices))
                bi = blocks_indices[i]
                mask_bi = block_bitmasks[bi]
                
                # Find critical t-subsets (covered only by bi)
                m = mask_bi
                critical_idxs = []
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cov[idx] == 1:
                        critical_idxs.append(idx)
                    m ^= lsb
                
                if len(critical_idxs) == 0:
                    # Should have been caught in Phase 1, but handle it
                    blocks_indices.pop(i)
                    m = mask_bi
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cov[idx] -= 1
                        m ^= lsb
                    improved = True
                    found_swap = True
                    break
                
                # Find replacement: must cover all critical t-subsets
                # Use the first critical t-subset to get candidates
                first_crit = critical_idxs[0]
                candidates = tsub_blocks[first_crit]
                
                best_replacement = -1
                best_new_gain = -1
                
                for bi2 in candidates:
                    if bi2 == bi:
                        continue
                    mask_bi2 = block_bitmasks[bi2]
                    
                    # Check if bi2 covers all critical subsets
                    covers_all = True
                    for ci in critical_idxs:
                        if not (mask_bi2 >> ci) & 1:
                            covers_all = False
                            break
                    
                    if not covers_all:
                        continue
                    
                    # Calculate net gain: new subsets covered by bi2 minus subsets lost from bi
                    # Net gain = |bi2 \ bi| - |bi \ bi2| but we only care about uncovered after swap
                    # Actually: after removing bi, some subsets become uncovered (those with cov==1 in bi)
                    # After adding bi2, those plus any new subsets covered by bi2 become covered
                    # Net change in coverage = (subsets in bi2 not in bi) - (subsets in bi not in bi2)
                    # But we need to check against current cov state
                    # Simpler: just count how many new subsets bi2 covers that aren't covered by bi
                    # AND ensure all critical are covered
                    
                    # For swap to be beneficial, we want bi2 to cover more "new" ground
                    # new_gain = number of subsets in bi2 that have cov > 0 (currently covered) 
                    #           minus those that would become uncovered
                    # Actually, the swap is valid if all critical are covered.
                    # The benefit is if bi2 covers subsets that bi doesn't, freeing up coverage
                    # But since we're replacing bi with bi2, the net effect on block count is 0.
                    # We do this swap to potentially enable future deletions.
                    # So we should prefer bi2 that covers more unique ground.
                    
                    # Count subsets covered by bi2 but not bi
                    diff_mask = mask_bi2 & ~mask_bi
                    gain2 = diff_mask.bit_count()
                    
                    # Count subsets covered by bi but not bi2
                    diff_mask_rev = mask_bi & ~mask_bi2
                    loss = diff_mask_rev.bit_count()
                    
                    net_gain = gain2 - loss
                    
                    if net_gain > best_new_gain:
                        best_new_gain = net_gain
                        best_replacement = bi2
                
                if best_replacement != -1:
                    # Perform the swap
                    old_mask = block_bitmasks[bi]
                    new_mask = block_bitmasks[best_replacement]
                    
                    # Remove old block's coverage
                    m = old_mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cov[idx] -= 1
                        m ^= lsb
                    
                    # Add new block's coverage
                    m = new_mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cov[idx] += 1
                        m ^= lsb
                    
                    blocks_indices[i] = best_replacement
                    improved = True
                    found_swap = True
                    break
            
            if not improved:
                break
        
        return blocks_indices
    
    def cover_holes(holes_mask):
        """Run greedy to cover only the holes in holes_mask."""
        new_blocks = []
        uncovered = holes_mask
        while uncovered:
            if time.time() > deadline:
                break
            best_idx = -1
            best_gain = 0
            for i in range(n_k_blocks):
                gain = (uncovered & block_bitmasks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            new_blocks.append(best_idx)
            uncovered &= ~block_bitmasks[best_idx]
        return new_blocks
    
    # Run initial greedy
    initial_blocks = greedy_cover()
    
    # Run local search
    current_blocks = local_search(initial_blocks)
    
    best_blocks = current_blocks
    best_count = len(current_blocks)
    
    # Perturbation-based local search
    while time.time() < deadline - 10:
        # Create perturbation
        perturbed_blocks = list(best_blocks)
        
        num_remove = min(len(perturbed_blocks), random.randint(3, 8))
        if num_remove == 0:
            break
        
        to_remove = random.sample(perturbed_blocks, num_remove)
        remove_set = set(to_remove)
        perturbed_blocks = [b for b in perturbed_blocks if b not in remove_set]
        
        # Calculate holes
        covered_mask = 0
        for b_idx in perturbed_blocks:
            covered_mask |= block_bitmasks[b_idx]
        
        holes_mask = ((1 << n_t_subsets) - 1) & ~covered_mask
        
        # Cover holes
        added_blocks = cover_holes(holes_mask)
        
        new_total_blocks = perturbed_blocks + added_blocks
        new_count = len(new_total_blocks)
        
        if new_count < best_count:
            cleaned_blocks = local_search(new_total_blocks)
            if len(cleaned_blocks) < best_count:
                best_blocks = cleaned_blocks
                best_count = len(best_blocks)
    
    # Final local search pass
    if time.time() < deadline - 2:
        final_blocks = local_search(best_blocks)
        if len(final_blocks) < best_count:
            best_blocks = final_blocks
            best_count = len(best_blocks)
    
    # Convert indices back to point lists
    final_blocks = [list(block_points[i]) for i in best_blocks]
    
    return final_blocks
