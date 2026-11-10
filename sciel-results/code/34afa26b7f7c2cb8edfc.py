# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def _build_precomputation(v, k, t):
    """
    Precompute all t-subsets, all k-blocks, and their coverage relationships.
    Returns:
        all_tsubs: list of all t-subsets (tuples)
        tsub_to_idx: dict mapping t-subset tuple to index
        n_tsubs: number of t-subsets
        all_kblocks: list of all k-blocks (tuples)
        block_to_idx: dict mapping k-block tuple to index
        n_kblocks: number of k-blocks
        block_covers: list of numpy arrays, block_covers[i] = indices of t-subsets covered by k-block i
        tsub_in_blocks: list of lists, tsub_in_blocks[j] = indices of k-blocks containing t-subset j
    """
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    all_kblocks = list(itertools.combinations(range(v), k))
    n_kblocks = len(all_kblocks)
    block_to_idx = {bk: i for i, bk in enumerate(all_kblocks)}
    
    block_covers = [None] * n_kblocks
    tsub_in_blocks = [[] for _ in range(n_tsubs)]
    
    for i, bk in enumerate(all_kblocks):
        covered = []
        for ts in itertools.combinations(bk, t):
            idx = tsub_to_idx[ts]
            covered.append(idx)
            tsub_in_blocks[idx].append(i)
        block_covers[i] = np.array(covered, dtype=np.int32)
    
    return all_tsubs, tsub_to_idx, n_tsubs, all_kblocks, block_to_idx, n_kblocks, block_covers, tsub_in_blocks


def _greedy_cover(v, k, t, rng, deadline, precomp, strategy='random'):
    """
    Greedy covering construction.
    strategy: 'random' - pick random uncovered t-subset, then best block.
              'bottleneck' - pick uncovered t-subset with fewest candidate blocks.
    """
    all_tsubs, tsub_to_idx, n_tsubs, all_kblocks, block_to_idx, n_kblocks, block_covers, tsub_in_blocks = precomp
    
    uncovered = np.ones(n_tsubs, dtype=bool)
    blocks_used = []
    
    # Precompute candidate counts for bottleneck strategy
    cand_counts = np.array([len(tsub_in_blocks[j]) for j in range(n_tsubs)], dtype=np.int32)
    
    while np.any(uncovered):
        if time.time() > deadline - 3.0:
            break
        
        unc_idx = np.where(uncovered)[0]
        if len(unc_idx) == 0:
            break
        
        if strategy == 'bottleneck':
            # Pick uncovered t-subset with minimum candidate count
            # Among uncovered, find the one with smallest cand_counts
            min_cand = np.inf
            best_target = unc_idx[0]
            # For speed, sample if too many
            if len(unc_idx) > 200:
                sample = rng.sample(unc_idx.tolist(), 200)
                for idx in sample:
                    if cand_counts[idx] < min_cand:
                        min_cand = cand_counts[idx]
                        best_target = idx
            else:
                for idx in unc_idx:
                    if cand_counts[idx] < min_cand:
                        min_cand = cand_counts[idx]
                        best_target = idx
        else:
            best_target = unc_idx[rng.randint(0, len(unc_idx) - 1)]
        
        candidates = tsub_in_blocks[best_target]
        if not candidates:
            break
        
        # Find block with maximum gain (covers most uncovered t-subsets)
        best_block = -1
        best_gain = -1
        for b_idx in candidates:
            gain = np.sum(uncovered[block_covers[b_idx]])
            if gain > best_gain:
                best_gain = gain
                best_block = b_idx
        
        if best_block == -1:
            break
        
        blocks_used.append(all_kblocks[best_block])
        uncovered[block_covers[best_block]] = False
    
    return blocks_used


def _remove_redundant(blocks, precomp):
    """
    Remove blocks that are redundant (all their t-subsets are covered by other blocks).
    """
    all_tsubs, tsub_to_idx, n_tsubs, all_kblocks, block_to_idx, n_kblocks, block_covers, tsub_in_blocks = precomp
    
    if not blocks:
        return blocks
    
    # Build coverage count
    cover_count = np.zeros(n_tsubs, dtype=np.int32)
    block_indices = []
    for block in blocks:
        b_idx = block_to_idx.get(tuple(block))
        if b_idx is None:
            # Should not happen, but handle gracefully
            b_idx = block_to_idx[tuple(sorted(block))]
        block_indices.append(b_idx)
        cover_count[block_covers[b_idx]] += 1
    
    new_blocks = []
    for i, b_idx in enumerate(block_indices):
        redundant = True
        for idx in block_covers[b_idx]:
            if cover_count[idx] <= 1:
                redundant = False
                break
        if not redundant:
            new_blocks.append(blocks[i])
        else:
            cover_count[block_covers[b_idx]] -= 1
    
    return new_blocks


def _local_search_polish(blocks, precomp, rng, deadline):
    """
    Local search: repeatedly remove redundant blocks and try to swap blocks
    to enable further removals.
    """
    all_tsubs, tsub_to_idx, n_tsubs, all_kblocks, block_to_idx, n_kblocks, block_covers, tsub_in_blocks = precomp
    
    current = list(blocks)
    improved = True
    
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # First pass: remove all redundancies
        current = _remove_redundant(current, precomp)
        
        # Try swapping blocks to enable further removal
        if time.time() > deadline - 8.0:
            break
        
        n = len(current)
        if n == 0:
            break
        
        # Build coverage count for current set
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        block_indices = []
        for block in current:
            b_idx = block_to_idx[tuple(block)]
            block_indices.append(b_idx)
            cover_count[block_covers[b_idx]] += 1
        
        # Try to find a block that can be swapped with another to reduce total
        rng.shuffle(range(n))
        for bi in range(n):
            if time.time() > deadline - 5.0:
                break
            
            b_idx = block_indices[bi]
            
            # Find t-subsets uniquely covered by this block
            unique_subs = []
            for idx in block_covers[b_idx]:
                if cover_count[idx] == 1:
                    unique_subs.append(idx)
            
            if not unique_subs:
                # This block is redundant - should have been removed
                continue
            
            # The replacement block must cover all unique_subs
            # Find candidate replacement blocks that cover all unique_subs
            # A block covers unique_subs if all those t-subsets are in block_covers[replacement]
            
            # For each unique t-subset, get the blocks containing it
            # The intersection of these sets gives candidates
            candidate_blocks = None
            for us_idx in unique_subs[:3]:  # Limit to first 3 for speed
                blocks_with = set(tsub_in_blocks[us_idx])
                if candidate_blocks is None:
                    candidate_blocks = blocks_with
                else:
                    candidate_blocks &= blocks_with
            
            if candidate_blocks is None or not candidate_blocks:
                continue
            
            # Exclude the current block and blocks already in the set
            current_block_set = set(block_indices)
            candidates = [cb for cb in candidate_blocks if cb not in current_block_set]
            
            if not candidates:
                continue
            
            # Try a few random candidates
            rng.shuffle(candidates)
            for cand_idx in candidates[:5]:
                # Check if adding cand_idx and removing bi still covers everything
                # After removal, check if any t-subset becomes uncovered
                new_cover = cover_count.copy()
                new_cover[block_covers[b_idx]] -= 1
                new_cover[block_covers[cand_idx]] += 1
                
                if np.any(new_cover < 0):
                    continue
                
                # This swap is valid. Perform it.
                current[bi] = all_kblocks[cand_idx]
                block_indices[bi] = cand_idx
                cover_count = new_cover
                improved = True
                break
        
        # Final redundancy removal
        current = _remove_redundant(current, precomp)
    
    return current


def _perturbation(blocks, precomp, rng, num_perturb=2):
    """
    Perturb the block set by removing and re-adding blocks.
    """
    all_tsubs, tsub_to_idx, n_tsubs, all_kblocks, block_to_idx, n_kblocks, block_covers, tsub_in_blocks = precomp
    
    if not blocks:
        return blocks
    
    current = list(blocks)
    n = len(current)
    
    # Remove num_perturb random blocks
    to_remove = rng.sample(range(n), min(num_perturb, n))
    to_remove_set = set(to_remove)
    current = [b for i, b in enumerate(current) if i not in to_remove_set]
    
    # Re-cover any uncovered t-subsets using greedy
    cover_count = np.zeros(n_tsubs, dtype=np.int32)
    for block in current:
        b_idx = block_to_idx[tuple(block)]
        cover_count[block_covers[b_idx]] += 1
    
    uncovered = cover_count == 0
    
    while np.any(uncovered):
        unc_idx = np.where(uncovered)[0]
        if len(unc_idx) == 0:
            break
        
        target = unc_idx[rng.randint(0, len(unc_idx) - 1)]
        candidates = tsub_in_blocks[target]
        if not candidates:
            break
        
        best_block = -1
        best_gain = -1
        for b_idx in candidates:
            gain = np.sum(uncovered[block_covers[b_idx]])
            if gain > best_gain:
                best_gain = gain
                best_block = b_idx
        
        if best_block == -1:
            break
        
        current.append(all_kblocks[best_block])
        uncovered[block_covers[best_block]] = False
    
    return current


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Build precomputation
    precomp = _build_precomputation(v, k, t)
    all_tsubs, tsub_to_idx, n_tsubs, all_kblocks, block_to_idx, n_kblocks, block_covers, tsub_in_blocks = precomp
    
    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Multiple greedy restarts with different strategies
    seeds = [42, 123, 7, 99, 2024, 555, 111, 222, 333, 444, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
             100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    strategies = ['random', 'bottleneck']
    
    for seed in seeds:
        if time.time() > deadline - 20.0:
            break
        
        rng = random.Random(seed)
        
        # Alternate strategies
        strategy = strategies[seed % len(strategies)]
        
        blocks = _greedy_cover(v, k, t, rng, deadline, precomp, strategy)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
        
        # Polish with local search
        if time.time() < deadline - 15.0:
            polished = _local_search_polish(blocks, precomp, rng, deadline)
            if len(polished) < best_count:
                best_count = len(polished)
                best_blocks = polished
    
    # Phase 2: Perturbation + local search on best solution
    if best_blocks is not None and time.time() < deadline - 10.0:
        for perturb_seed in [42, 123, 7, 99, 2024]:
            if time.time() > deadline - 5.0:
                break
            
            rng_p = random.Random(perturb_seed)
            perturbed = _perturbation(best_blocks, precomp, rng_p, num_perturb=2)
            
            if len(perturbed) < best_count:
                best_count = len(perturbed)
                best_blocks = perturbed
            
            polished = _local_search_polish(perturbed, precomp, rng_p, deadline)
            if len(polished) < best_count:
                best_count = len(polished)
                best_blocks = polished
    
    # Final: ensure we return a valid covering
    if best_blocks is None:
        rng = random.Random(0)
        best_blocks = _greedy_cover(v, k, t, rng, deadline, precomp, 'random')
    
    # Convert to lists of integers
    result = [list(block) for block in best_blocks]
    
    return result
# EVOLVE-BLOCK-END
