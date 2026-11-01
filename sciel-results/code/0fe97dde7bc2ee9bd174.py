import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit
    
    # Precompute t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    
    # Precompute k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    # Build bitmasks for each block
    block_bitmasks = [0] * n_k_blocks
    for i, block in enumerate(all_k_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            mask |= (1 << idx)
        block_bitmasks[i] = mask
    
    # Build inverted index: for each t-subset, which blocks cover it
    # This enables faster local search operations
    t_subset_to_blocks = defaultdict(list)
    for i, mask in enumerate(block_bitmasks):
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            t_subset_to_blocks[idx].append(i)
            m ^= lsb
    
    full_mask = (1 << n_t_subsets) - 1
    
    def greedy_cover(uncovered_mask=None):
        """Greedy set cover starting from a given uncovered mask."""
        if uncovered_mask is None:
            uncovered_mask = full_mask
        
        selected_blocks = []
        # Track coverage count for each t-subset to enable fast removal later
        coverage_counts = np.zeros(n_t_subsets, dtype=np.int32)
        
        while uncovered_mask:
            if time.time() > deadline:
                break
            
            best_idx = -1
            best_gain = 0
            
            # Find block with max marginal gain
            for i in range(n_k_blocks):
                gain = (uncovered_mask & block_bitmasks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx == -1:
                break
            
            selected_blocks.append(best_idx)
            
            # Update coverage counts and uncovered mask
            mask = block_bitmasks[best_idx]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                coverage_counts[idx] += 1
                m ^= lsb
            
            uncovered_mask &= ~block_bitmasks[best_idx]
        
        return selected_blocks, coverage_counts
    
    def local_search_remove(blocks_indices, coverage_counts):
        """Remove redundant blocks iteratively."""
        if time.time() > deadline:
            return blocks_indices, coverage_counts
        
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            to_remove = []
            
            for b_idx in blocks_indices:
                mask = block_bitmasks[b_idx]
                m = mask
                essential = False
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if coverage_counts[idx] == 1:
                        essential = True
                        break
                    m ^= lsb
                if not essential:
                    to_remove.append(b_idx)
            
            if to_remove:
                remove_set = set(to_remove)
                blocks_indices = [b for b in blocks_indices if b not in remove_set]
                for b_idx in to_remove:
                    mask = block_bitmasks[b_idx]
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        coverage_counts[idx] -= 1
                        m ^= lsb
                changed = True
        
        return blocks_indices, coverage_counts
    
    def local_search_swap(blocks_indices, coverage_counts):
        """Try to swap blocks to reduce count: remove one, add one that covers more."""
        if time.time() > deadline:
            return blocks_indices, coverage_counts
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            for i, b_idx in enumerate(blocks_indices):
                if time.time() > deadline:
                    break
                
                # Temporarily remove b_idx
                mask_to_remove = block_bitmasks[b_idx]
                m = mask_to_remove
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    coverage_counts[idx] -= 1
                    m ^= lsb
                
                # Find uncovered t-subsets
                uncovered_mask = 0
                for j in range(n_t_subsets):
                    if coverage_counts[j] == 0:
                        uncovered_mask |= (1 << j)
                
                if uncovered_mask == 0:
                    # Can remove b_idx entirely
                    blocks_indices = [b for j, b in enumerate(blocks_indices) if j != i]
                    improved = True
                    break
                
                # Try to find a block that covers the most uncovered (vectorized)
                # Build a numpy boolean array of which blocks are currently selected
                selected_set = set(blocks_indices)
                # Compute gains for all blocks using numpy on precomputed block-t-subset incidence
                # We need a faster way: use the precomputed t_subset_to_blocks index
                # Count how many uncovered t-subsets each candidate block covers
                # Build a sparse-like approach: for each uncovered t-subset, add 1 to all blocks covering it
                uncovered_count = np.zeros(n_k_blocks, dtype=np.int32)
                unc_m = uncovered_mask
                while unc_m:
                    lsb = unc_m & -unc_m
                    ts_idx = lsb.bit_length() - 1
                    for blk in t_subset_to_blocks[ts_idx]:
                        if blk not in selected_set:
                            uncovered_count[blk] += 1
                    unc_m ^= lsb
                
                best_new_idx = -1
                best_new_gain = 0
                # Find max gain among non-selected blocks
                for blk in range(n_k_blocks):
                    if blk in selected_set:
                        continue
                    g = uncovered_count[blk]
                    if g > best_new_gain:
                        best_new_gain = g
                        best_new_idx = blk
                
                # Restore b_idx
                m = mask_to_remove
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    coverage_counts[idx] += 1
                    m ^= lsb
                
                # If we can cover all uncovered with one new block, do the swap
                if best_new_idx != -1 and best_new_gain >= uncovered_mask.bit_count():
                    # Remove old block
                    m = mask_to_remove
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        coverage_counts[idx] -= 1
                        m ^= lsb
                    
                    blocks_indices[i] = best_new_idx
                    
                    # Add new block
                    mask_to_add = block_bitmasks[best_new_idx]
                    m = mask_to_add
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        coverage_counts[idx] += 1
                        m ^= lsb
                    
                    improved = True
                    break
        
        return blocks_indices, coverage_counts
    
    # Initial greedy
    initial_blocks, initial_counts = greedy_cover()
    
    # Local search: remove then swap
    current_blocks, current_counts = local_search_remove(initial_blocks, initial_counts)
    current_blocks, current_counts = local_search_swap(current_blocks, current_counts)
    
    best_blocks = list(current_blocks)
    best_counts = current_counts.copy()
    best_count = len(current_blocks)
    
    # Perturbation-based local search
    def compute_coverage_mask(blocks_list):
        """Compute the coverage mask for a list of block indices."""
        mask = 0
        for b_idx in blocks_list:
            mask |= block_bitmasks[b_idx]
        return mask
    
    def cover_holes_greedy(existing_blocks, holes_mask):
        """Cover holes using greedy, returning new blocks added."""
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
    
    def rebuild_counts(blocks_list):
        """Rebuild coverage counts from scratch."""
        counts = np.zeros(n_t_subsets, dtype=np.int32)
        for b_idx in blocks_list:
            mask = block_bitmasks[b_idx]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
        return counts
    
    # Perturbation attempts
    attempts = 0
    while time.time() < deadline - 10:
        attempts += 1
        
        # Create perturbation: remove 3-8 random blocks
        num_remove = min(len(best_blocks), random.randint(3, 8))
        if num_remove == 0:
            break
        
        to_remove = random.sample(best_blocks, num_remove)
        remove_set = set(to_remove)
        perturbed_blocks = [b for b in best_blocks if b not in remove_set]
        
        # Compute holes
        covered_mask = compute_coverage_mask(perturbed_blocks)
        holes_mask = full_mask & ~covered_mask
        
        # Cover holes
        added_blocks = cover_holes_greedy(perturbed_blocks, holes_mask)
        
        new_total_blocks = perturbed_blocks + added_blocks
        new_count = len(new_total_blocks)
        
        if new_count < best_count:
            # Rebuild counts and run local search
            new_counts = rebuild_counts(new_total_blocks)
            cleaned_blocks, cleaned_counts = local_search_remove(new_total_blocks, new_counts)
            cleaned_blocks, cleaned_counts = local_search_swap(cleaned_blocks, cleaned_counts)
            
            if len(cleaned_blocks) < best_count:
                best_blocks = cleaned_blocks
                best_counts = cleaned_counts
                best_count = len(best_blocks)
            elif len(new_total_blocks) < best_count:
                best_blocks = new_total_blocks
                best_counts = new_counts
                best_count = len(best_blocks)
    
    # Final local search pass
    if time.time() < deadline - 5:
        best_blocks, best_counts = local_search_remove(best_blocks, best_counts)
        best_blocks, best_counts = local_search_swap(best_blocks, best_counts)
    
    # Simulated annealing polish: try single-block swaps with temperature-based acceptance
    if time.time() < deadline - 2:
        sa_blocks = list(best_blocks)
        sa_counts = best_counts.copy()
        sa_count = len(sa_blocks)
        sa_best = sa_count
        sa_best_blocks = list(sa_blocks)
        sa_best_counts = sa_counts.copy()
        
        # Precompute for SA: for each block, the list of t-subsets it covers
        block_t_subsets = []
        for i in range(n_k_blocks):
            mask = block_bitmasks[i]
            ts_list = []
            m = mask
            while m:
                lsb = m & -m
                ts_list.append(lsb.bit_length() - 1)
                m ^= lsb
            block_t_subsets.append(ts_list)
        
        # SA parameters
        sa_start_temp = 3.0
        sa_end_temp = 0.1
        sa_iterations = 0
        sa_max_iter = 5000
        
        while time.time() < deadline - 2 and sa_iterations < sa_max_iter:
            sa_iterations += 1
            temp = sa_start_temp * (sa_end_temp / sa_start_temp) ** (sa_iterations / sa_max_iter)
            
            if len(sa_blocks) == 0:
                break
            
            # Randomly pick a block to remove and a candidate to add
            old_idx_pos = random.randint(0, len(sa_blocks) - 1)
            old_blk = sa_blocks[old_idx_pos]
            
            # Find a random candidate block not in current set
            candidate = random.randint(0, n_k_blocks - 1)
            if candidate in sa_blocks:
                continue
            
            # Compute delta: how many newly covered vs newly uncovered t-subsets
            # Remove old_blk: t-subsets with count==1 become uncovered
            # Add candidate: t-subsets with count==0 become covered
            
            old_ts = block_t_subsets[old_blk]
            cand_ts = block_t_subsets[candidate]
            
            newly_uncovered = 0
            newly_covered = 0
            
            for ts in old_ts:
                if sa_counts[ts] == 1:
                    newly_uncovered += 1
            
            for ts in cand_ts:
                if sa_counts[ts] == 0:
                    newly_covered += 1
            
            # Net change in uncovered count
            delta_uncovered = newly_uncovered - newly_covered
            
            # Acceptance: always accept if delta <= 0 (improvement or neutral)
            # Otherwise accept with probability exp(-delta/temp)
            if delta_uncovered <= 0:
                accept = True
            else:
                accept = random.random() < np.exp(-delta_uncovered / max(temp, 1e-10))
            
            if accept:
                # Apply the swap
                for ts in old_ts:
                    sa_counts[ts] -= 1
                for ts in cand_ts:
                    sa_counts[ts] += 1
                sa_blocks[old_idx_pos] = candidate
                
                if len(sa_blocks) < sa_best:
                    sa_best = len(sa_blocks)
                    sa_best_blocks = list(sa_blocks)
                    sa_best_counts = sa_counts.copy()
            else:
                # Don't apply, just continue
                pass
        
        # Run local search on the best SA result
        if sa_best < best_count:
            cleaned_blocks, cleaned_counts = local_search_remove(sa_best_blocks, sa_best_counts)
            cleaned_blocks, cleaned_counts = local_search_swap(cleaned_blocks, cleaned_counts)
            if len(cleaned_blocks) < best_count:
                best_blocks = cleaned_blocks
                best_counts = cleaned_counts
                best_count = len(best_blocks)
            elif sa_best < best_count:
                best_blocks = sa_best_blocks
                best_counts = sa_best_counts
                best_count = sa_best
    
    # Convert to output format
    final_blocks = [list(all_k_blocks[i]) for i in best_blocks]
    
    return final_blocks
