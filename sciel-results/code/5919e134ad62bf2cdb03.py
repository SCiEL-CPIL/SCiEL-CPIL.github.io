"""Weighted greedy covering with local search refinement and deadline guard."""
import itertools
import time
import numpy as np
import random
import math

def construct(v, k, t):
    deadline = time.monotonic() + 115.0  # safety margin under 120s
    points = list(range(v))
    
    # Generate all t-subsets and assign them indices for fast lookup
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    # Precompute all possible k-blocks
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)
    
    # For each k-block, precompute the list of t-subset indices it covers
    block_tsubs = []
    for b in all_kblocks:
        b_tsubs = [tsub_to_idx[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(b_tsubs)
    
    # Build coverage matrix: coverage[i, j] = 1 if block i covers t-subset j
    t_per_block = len(block_tsubs[0])
    coverage = np.zeros((num_kblocks, num_tsubs), dtype=np.int8)
    for i, bts in enumerate(block_tsubs):
        for j in bts:
            coverage[i, j] = 1
    
    # Compute frequency of each t-subset (how many blocks cover it)
    subset_frequencies = coverage.sum(axis=0)
    
    # Weight vectors for different greedy strategies
    weights_inv = np.where(subset_frequencies > 0, 1.0 / subset_frequencies, 0.0).astype(np.float64)
    weights_uniform = np.ones(num_tsubs, dtype=np.float64)
    weights_sqrt_inv = np.where(subset_frequencies > 0, 1.0 / np.sqrt(subset_frequencies), 0.0).astype(np.float64)
    
    def weighted_greedy(weights, rng=None):
        """Run weighted greedy covering."""
        uncovered = np.ones(num_tsubs, dtype=np.int8)
        remaining = num_tsubs
        chosen = []
        selected = np.zeros(num_kblocks, dtype=bool)
        
        while remaining > 0:
            if time.monotonic() > deadline:
                break
            
            # Compute weighted gain for each block
            weighted_uncovered = weights * uncovered
            gains = coverage @ weighted_uncovered
            
            # Exclude already selected blocks
            gains[selected] = -1
            
            max_gain = gains.max()
            if max_gain <= 0:
                break
            
            best_indices = np.where(gains == max_gain)[0]
            
            if len(best_indices) == 1 or rng is None:
                best_idx = int(best_indices[0])
            else:
                best_idx = int(best_indices[rng.randint(0, len(best_indices) - 1)])
            
            chosen.append(best_idx)
            selected[best_idx] = True
            
            # Update uncovered
            newly_covered = coverage[best_idx]
            still_uncovered = uncovered & (1 - newly_covered)
            remaining = int(still_uncovered.sum())
            uncovered = still_uncovered
        
        return chosen
    
    def local_search(block_indices, deadline):
        """Remove redundant blocks and try 1-for-1 swaps to reduce count."""
        current = list(block_indices)
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for idx in current:
            counts[coverage[idx] == 1] += 1
        
        best_len = len(current)
        best_blocks = current[:]
        
        no_improve = 0
        max_no_improve = 50
        
        while time.monotonic() < deadline and no_improve < max_no_improve:
            # Phase A: Remove all redundant blocks
            progress = True
            while progress:
                progress = False
                for pos in range(len(current)):
                    idx = current[pos]
                    # Check if any t-subset is only covered by this block
                    mask = coverage[idx] == 1
                    if not (counts[mask] == 1).any():
                        # This block is redundant
                        counts[mask] -= 1
                        current.pop(pos)
                        progress = True
                        if len(current) < best_len:
                            best_len = len(current)
                            best_blocks = current[:]
                        break
            
            if len(current) == 0:
                break
            
            # Phase B: Try 1-for-1 swaps
            swap_found = False
            n_sample = min(len(current), 50)
            sampled = random.sample(range(len(current)), n_sample)
            
            for i in sampled:
                if time.monotonic() > deadline:
                    break
                idx_i = current[i]
                mask_i = coverage[idx_i] == 1
                unique_to_i = mask_i & (counts == 1)
                n_unique = int(unique_to_i.sum())
                
                if n_unique == 0:
                    # Block is redundant, remove it
                    counts[mask_i] -= 1
                    current.pop(i)
                    swap_found = True
                    if len(current) < best_len:
                        best_len = len(current)
                        best_blocks = current[:]
                    break
                
                if n_unique < 3:
                    # Try to find a replacement block that covers the unique subsets
                    # and adds minimal new coverage
                    unique_mask = unique_to_i.astype(np.int8)
                    # Find blocks that cover all unique subsets
                    candidate_gains = coverage @ (unique_mask * 2 - coverage @ unique_mask)
                    # Actually, let's find blocks that cover the unique subsets
                    # and have low overlap with current coverage
                    
                    # Compute for each candidate block: how many of idx_i's unique subsets it covers
                    covers_unique = (coverage @ unique_mask)
                    # We want blocks that cover all unique subsets
                    good_candidates = np.where(covers_unique == n_unique)[0]
                    
                    if len(good_candidates) > 0:
                        # Among good candidates, pick one with minimum overlap with current
                        # (i.e., adds the fewest new covered subsets)
                        current_coverage = np.zeros(num_tsubs, dtype=np.int8)
                        for j in current:
                            current_coverage |= coverage[j]
                        
                        overlap = (coverage[good_candidates] & current_coverage).sum(axis=1)
                        best_candidate = int(good_candidates[np.argmin(overlap)])
                        
                        # Perform the swap
                        counts[mask_i] -= 1
                        mask_new = coverage[best_candidate] == 1
                        counts[mask_new] += 1
                        current[i] = best_candidate
                        swap_found = True
                        break
            
            if swap_found:
                no_improve = 0
            else:
                no_improve += 1
        
        return best_blocks
    
    # Run multiple weighted greedy strategies
    best_blocks = None
    best_len = float('inf')
    
    strategies = [
        (weights_inv, None),
        (weights_uniform, None),
        (weights_sqrt_inv, None),
    ]
    
    for weights, seed in strategies:
        if time.monotonic() > deadline - 10:
            break
        
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = None
        
        blocks = weighted_greedy(weights, rng)
        
        if blocks and len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]
    
    # Try a few randomized restarts with different seeds
    for attempt in range(1, 8):
        if time.monotonic() > deadline - 15:
            break
        
        seed = 100 + attempt
        rng = random.Random(seed)
        weights = weights_inv if attempt % 2 == 0 else weights_uniform
        blocks = weighted_greedy(weights, rng)
        
        if blocks and len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]
    
    # Apply local search to the best result
    if best_blocks is not None and time.monotonic() < deadline - 5:
        refined = local_search(best_blocks, deadline - 2)
        if len(refined) < best_len:
            best_blocks = refined
            best_len = len(refined)
    
    # Convert block indices back to actual blocks
    result = [list(all_kblocks[idx]) for idx in best_blocks]
    
    return result
