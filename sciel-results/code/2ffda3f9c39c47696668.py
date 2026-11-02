import itertools
import random
import time
import numpy as np

# EVOLVE-BLOCK-START

def _precompute(v, k, t):
    """Precompute all t-sets, all k-blocks, and bitset representations."""
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    
    # Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, compute which t-sets it covers (as a set of tset indices)
    block_tsets = []
    for block in all_blocks:
        tset_indices = set()
        for ts in itertools.combinations(block, t):
            tset_indices.add(tset_to_idx[ts])
        block_tsets.append(tset_indices)
    
    # Precompute tset_to_block_indices: for each tset, which blocks cover it
    tset_to_block_indices = [[] for _ in range(n_tsets)]
    for b_idx in range(n_blocks):
        for ts_idx in block_tsets[b_idx]:
            tset_to_block_indices[ts_idx].append(b_idx)
    
    # Build bitset representations
    # block_masks[b_idx] is a Python int with bit i set iff tset i is covered by block b_idx
    block_masks = []
    for tset_set in block_tsets:
        mask = 0
        for ts_idx in tset_set:
            mask |= (1 << ts_idx)
        block_masks.append(mask)
    
    # full_mask: all n_tsets bits set
    full_mask = (1 << n_tsets) - 1
    
    return all_tsets, n_tsets, tset_to_idx, all_blocks, n_blocks, block_tsets, tset_to_block_indices, block_masks, full_mask

def _greedy_covering(n_tsets, n_blocks, block_tsets, tset_to_block_indices, block_masks, full_mask, rng, time_limit):
    """Fast greedy set cover using bitset operations."""
    deadline = time.time() + time_limit
    uncovered_mask = full_mask
    blocks_chosen = []
    
    while uncovered_mask != 0 and time.time() < deadline:
        # Pick a random uncovered t-set
        # Find one set bit in uncovered_mask
        # Use lowest set bit
        lsb = uncovered_mask & (-uncovered_mask)
        rand_idx = lsb.bit_length() - 1
        
        # Find best block covering this t-set (and maximizing total uncovered coverage)
        best_block = -1
        best_gain = -1
        for b_idx in tset_to_block_indices[rand_idx]:
            gain = bin(block_masks[b_idx] & uncovered_mask).count('1')
            if gain > best_gain:
                best_gain = gain
                best_block = b_idx
        
        if best_block == -1 or best_gain <= 0:
            break
            
        blocks_chosen.append(best_block)
        uncovered_mask &= ~block_masks[best_block]
        
    return blocks_chosen

def _local_search_bitset(blocks_idx, n_tsets, n_blocks, block_masks, full_mask, tset_to_block_indices, rng, time_limit):
    """Local search using bitset operations for O(1) marginal gain evaluation."""
    deadline = time.time() + time_limit
    current = list(blocks_idx)
    
    # Compute current coverage mask
    cov_mask = 0
    for b_idx in current:
        cov_mask |= block_masks[b_idx]
    
    current_set = set(current)
    uncovered_count = n_tsets - bin(cov_mask).count('1')
    best = current[:]
    best_uncovered = uncovered_count
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try removing redundant blocks
        for i in range(len(current)):
            if time.time() > deadline:
                break
            b_idx = current[i]
            # Check if removing this block leaves everything covered
            if (cov_mask | block_masks[b_idx]) == cov_mask:
                # Block is redundant
                current.pop(i)
                current_set.discard(b_idx)
                cov_mask &= ~block_masks[b_idx]
                uncovered_count = n_tsets - bin(cov_mask).count('1')
                if uncovered_count < best_uncovered:
                    best_uncovered = uncovered_count
                    best = current[:]
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: Try 1-swap: replace one block with another
        for i in range(len(current)):
            if time.time() > deadline:
                break
            old_b = current[i]
            old_mask = block_masks[old_b]
            
            # T-sets that become uncovered if we remove old_b
            newly_uncovered_mask = old_mask & ~cov_mask  # bits only covered by old_b
            
            if newly_uncovered_mask == 0:
                # Just remove it (redundant)
                current.pop(i)
                current_set.discard(old_b)
                cov_mask &= ~old_mask
                uncovered_count = n_tsets - bin(cov_mask).count('1')
                if uncovered_count < best_uncovered:
                    best_uncovered = uncovered_count
                    best = current[:]
                improved = True
                break
            
            # Try to find a replacement block that covers all newly_uncovered
            # and maximizes additional coverage
            best_new_b = -1
            best_gain = -1
            
            # Candidate blocks must cover all newly_uncovered t-sets
            # Start with candidates from the first newly_uncovered t-set
            lsb = newly_uncovered_mask & (-newly_uncovered_mask)
            first_ts_idx = lsb.bit_length() - 1
            
            for b_idx in tset_to_block_indices[first_ts_idx]:
                if b_idx == old_b or b_idx in current_set:
                    continue
                new_mask = block_masks[b_idx]
                # Check if new block covers all newly_uncovered
                if (new_mask & newly_uncovered_mask) != newly_uncovered_mask:
                    continue
                # Count additional coverage (new t-sets covered)
                gain = bin(new_mask & ~cov_mask).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_new_b = b_idx
            
            if best_new_b != -1:
                # Perform swap
                current[i] = best_new_b
                current_set.discard(old_b)
                current_set.add(best_new_b)
                cov_mask = (cov_mask & ~old_mask) | block_masks[best_new_b]
                uncovered_count = n_tsets - bin(cov_mask).count('1')
                if uncovered_count < best_uncovered:
                    best_uncovered = uncovered_count
                    best = current[:]
                improved = True
                break
        
        if not improved:
            # Phase 3: Try 2-swap: replace two blocks with two different blocks
            if len(current) >= 2:
                found_2swap = False
                for i in range(len(current)):
                    if time.time() > deadline or found_2swap:
                        break
                    for j in range(i+1, len(current)):
                        if time.time() > deadline or found_2swap:
                            break
                        old_i = current[i]
                        old_j = current[j]
                        old_mask_i = block_masks[old_i]
                        old_mask_j = block_masks[old_j]
                        
                        # Combined mask of the two blocks
                        combined_old = old_mask_i | old_mask_j
                        # T-sets that become uncovered
                        newly_uncovered_mask = combined_old & ~cov_mask
                        
                        if newly_uncovered_mask == 0:
                            # Both are redundant together
                            current.pop(j)
                            current.pop(i)
                            current_set.discard(old_i)
                            current_set.discard(old_j)
                            cov_mask &= ~combined_old
                            uncovered_count = n_tsets - bin(cov_mask).count('1')
                            if uncovered_count < best_uncovered:
                                best_uncovered = uncovered_count
                                best = current[:]
                            improved = True
                            found_2swap = True
                            break
                        
                        # Try to find two replacement blocks
                        # This is expensive, so limit candidates
                        lsb = newly_uncovered_mask & (-newly_uncovered_mask)
                        first_ts_idx = lsb.bit_length() - 1
                        candidates = tset_to_block_indices[first_ts_idx]
                        
                        # Limit candidate list for speed
                        if len(candidates) > 200:
                            candidates = rng.sample(candidates, 200)
                        
                        best_pair = None
                        best_pair_gain = -1
                        
                        for ci in range(len(candidates)):
                            if time.time() > deadline:
                                break
                            b1 = candidates[ci]
                            if b1 == old_i or b1 == old_j or b1 in current_set:
                                continue
                            mask1 = block_masks[b1]
                            # Check if b1 covers some of newly_uncovered
                            if (mask1 & newly_uncovered_mask) == 0:
                                continue
                            remaining_after_b1 = newly_uncovered_mask & ~mask1
                            if remaining_after_b1 == 0:
                                # b1 alone covers all, look for a second block for extra gain
                                # Just try adding one more block for max gain
                                for b2 in candidates:
                                    if b2 == b1 or b2 == old_i or b2 == old_j or b2 in current_set:
                                        continue
                                    mask2 = block_masks[b2]
                                    gain = bin((mask1 | mask2) & ~cov_mask).count('1')
                                    if gain > best_pair_gain:
                                        best_pair_gain = gain
                                        best_pair = (b1, b2)
                                if best_pair is not None:
                                    break
                            else:
                                # Need b2 to cover remaining_after_b1
                                lsb2 = remaining_after_b1 & (-remaining_after_b1)
                                second_ts_idx = lsb2.bit_length() - 1
                                for b2 in tset_to_block_indices[second_ts_idx]:
                                    if b2 == b1 or b2 == old_i or b2 == old_j or b2 in current_set:
                                        continue
                                    mask2 = block_masks[b2]
                                    if (mask2 & remaining_after_b1) != remaining_after_b1:
                                        continue
                                    gain = bin((mask1 | mask2) & ~cov_mask).count('1')
                                    if gain > best_pair_gain:
                                        best_pair_gain = gain
                                        best_pair = (b1, b2)
                                    if time.time() > deadline:
                                        break
                        
                        if best_pair is not None and best_pair_gain > 0:
                            b1, b2 = best_pair
                            new_combined = block_masks[b1] | block_masks[b2]
                            current[i] = b1
                            current[j] = b2
                            current_set.discard(old_i)
                            current_set.discard(old_j)
                            current_set.add(b1)
                            current_set.add(b2)
                            cov_mask = (cov_mask & ~combined_old) | new_combined
                            uncovered_count = n_tsets - bin(cov_mask).count('1')
                            if uncovered_count < best_uncovered:
                                best_uncovered = uncovered_count
                                best = current[:]
                            improved = True
                            found_2swap = True
                            break
            
            if improved:
                continue
            
            # Phase 4: Random perturbation to escape local optima
            if len(current) >= 2:
                for _ in range(30):
                    if time.time() > deadline:
                        break
                    i, j = rng.sample(range(len(current)), 2)
                    old_i, old_j = current[i], current[j]
                    
                    # Remove both
                    combined_old = block_masks[old_i] | block_masks[old_j]
                    cov_mask &= ~combined_old
                    current_set.discard(old_i)
                    current_set.discard(old_j)
                    
                    # Pick two random blocks
                    new_i, new_j = rng.sample(range(n_blocks), 2)
                    current[i] = new_i
                    current[j] = new_j
                    current_set.add(new_i)
                    current_set.add(new_j)
                    cov_mask |= block_masks[new_i] | block_masks[new_j]
                    
                    uncovered_count = n_tsets - bin(cov_mask).count('1')
                    if uncovered_count < best_uncovered:
                        best_uncovered = uncovered_count
                        best = current[:]
    
    return best

def _verify_covering(blocks_idx, block_masks, full_mask):
    """Verify that the blocks cover all t-sets using bitset."""
    cov_mask = 0
    for b_idx in blocks_idx:
        cov_mask |= block_masks[b_idx]
    return cov_mask == full_mask

def construct(v, k, t):
    start_time = time.time()
    total_budget = 145.0  # Safety margin before 150s limit
    
    # Precompute
    (all_tsets, n_tsets, tset_to_idx, all_blocks, n_blocks, 
     block_tsets, tset_to_block_indices, block_masks, full_mask) = _precompute(v, k, t)
    
    # Multiple greedy runs with different seeds
    best_blocks = None
    best_size = float('inf')
    
    for seed in range(10):
        if time.time() - start_time > total_budget - 60.0:
            break
        rng = random.Random(seed)
        blocks_idx = _greedy_covering(n_tsets, n_blocks, block_tsets, tset_to_block_indices, 
                                       block_masks, full_mask, rng, 5.0)
        
        if len(blocks_idx) < best_size:
            best_size = len(blocks_idx)
            best_blocks = blocks_idx[:]
    
    if best_blocks is None:
        # Fallback: just return first greedy result
        rng = random.Random(42)
        best_blocks = _greedy_covering(n_tsets, n_blocks, block_tsets, tset_to_block_indices, 
                                        block_masks, full_mask, rng, 10.0)
        best_size = len(best_blocks)
    
    # Local search on best solution using bitset
    if time.time() - start_time < total_budget - 10.0:
        rng = random.Random(123)
        remaining_time = min(80.0, total_budget - (time.time() - start_time) - 10.0)
        improved = _local_search_bitset(best_blocks, n_tsets, n_blocks, block_masks, full_mask, 
                                         tset_to_block_indices, rng, remaining_time)
        if len(improved) < best_size:
            best_blocks = improved
            best_size = len(improved)
    
    # Try to reduce further with more aggressive local search
    if time.time() - start_time < total_budget - 10.0 and best_size > 70:
        rng = random.Random(456)
        remaining_time = min(30.0, total_budget - (time.time() - start_time) - 10.0)
        improved = _local_search_bitset(best_blocks, n_tsets, n_blocks, block_masks, full_mask, 
                                         tset_to_block_indices, rng, remaining_time)
        if len(improved) < best_size:
            best_blocks = improved
            best_size = len(improved)
    
    # Final verification
    if not _verify_covering(best_blocks, block_masks, full_mask):
        # Something went wrong; rebuild
        rng = random.Random(789)
        best_blocks = _greedy_covering(n_tsets, n_blocks, block_tsets, tset_to_block_indices, 
                                        block_masks, full_mask, rng, 10.0)
    
    # Convert to list of lists
    result = [list(all_blocks[idx]) for idx in best_blocks]
    
    return result

# EVOLVE-BLOCK-END
