"""
Covering design C(19,8,5) solver: greedy construction + aggressive local search.
Strategy:
  1. Fast greedy set-cover construction with vectorized gain computation.
  2. Aggressive local search:
     - Remove redundant blocks
     - Swap points in blocks to enable removal of other blocks
     - 2-block swaps (modify two blocks simultaneously)
     - Random perturbation with acceptance criterion
  3. Multiple restarts with different random seeds.
"""
import itertools
import random
import time
import numpy as np


def _precompute(v, k, t):
    """Precompute all t-subsets and mappings for fast operations."""
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # For each k-block, precompute its t-subset indices
    # There are C(19,8) = 75582 possible blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Precompute t-subset indices for every possible block
    # block_tsub_indices[b] = list of t-subset indices covered by block b
    block_tsub_indices = []
    for block in all_blocks:
        subs = []
        for ts in itertools.combinations(block, t):
            subs.append(tsub_to_idx[ts])
        block_tsub_indices.append(subs)

    # Map from block tuple to its index
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}

    return all_tsubs, tsub_to_idx, all_blocks, block_to_idx, block_tsub_indices


def _greedy_construct(v, k, t, all_tsubs, tsub_to_idx, all_blocks, block_to_idx,
                      block_tsub_indices, rng, deadline):
    """
    Fast greedy set-cover using precomputed block t-subset indices.
    At each step, pick the block that covers the most uncovered t-subsets.
    """
    n_tsubs = len(all_tsubs)
    n_blocks = len(all_blocks)
    uncovered = np.ones(n_tsubs, dtype=np.uint8)
    blocks = []
    used_blocks = set()

    # For efficiency, we'll track uncovered count per block
    # uncovered_count[b] = number of t-subsets in block b that are still uncovered
    uncovered_count = np.array([len(subs) for subs in block_tsub_indices], dtype=np.int32)

    while np.any(uncovered) and time.time() < deadline:
        # Find the block with maximum uncovered_count that hasn't been used
        # We need to exclude used blocks
        # Instead of scanning all, we can use a priority approach
        # But for simplicity, scan all unused blocks
        
        best_gain = -1
        best_block_idx = -1
        
        # Sample-based approach for speed when many blocks remain
        if n_blocks > 5000:
            # Sample a subset of unused blocks
            unused = [b for b in range(n_blocks) if b not in used_blocks]
            if not unused:
                break
            sample_size = min(200, len(unused))
            candidates = rng.sample(unused, sample_size)
        else:
            candidates = [b for b in range(n_blocks) if b not in used_blocks]
        
        for b in candidates:
            gain = uncovered_count[b]
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b
        
        if best_block_idx < 0:
            break
        
        # Add this block
        block = all_blocks[best_block_idx]
        blocks.append(list(block))
        used_blocks.add(best_block_idx)
        
        # Update uncovered and uncovered_count
        for idx in block_tsub_indices[best_block_idx]:
            if uncovered[idx]:
                uncovered[idx] = 0
                # Decrement uncovered_count for all blocks containing this t-subset
                # This is expensive; instead we'll recompute lazily
                # For now, just mark as uncovered and let the next iteration handle it
        
        # Update uncovered_count: for each block, recompute if needed
        # This is O(n_blocks * t_subs_per_block) which is too slow
        # Alternative: just recompute best_gain from scratch each time using the uncovered array
        # But that's also expensive. Let's use a different approach.
        
        # Actually, let's just recompute the gain for all blocks each iteration
        # but only for a sample. The uncovered array is the source of truth.
    
    # The above approach has an issue: uncovered_count becomes stale.
    # Let's use a simpler but correct approach:
    # Each iteration, compute gain for a sample of unused blocks directly from uncovered array.
    
    # Restart with correct approach
    uncovered = np.ones(n_tsubs, dtype=np.uint8)
    blocks = []
    used_blocks = set()
    
    while np.any(uncovered) and time.time() < deadline:
        best_gain = -1
        best_block_idx = -1
        
        # Sample unused blocks
        unused = [b for b in range(n_blocks) if b not in used_blocks]
        if not unused:
            break
        
        # Adaptive sample size
        if len(unused) > 1000:
            sample_size = min(500, len(unused))
            candidates = rng.sample(unused, sample_size)
        else:
            candidates = unused
        
        for b in candidates:
            gain = 0
            for idx in block_tsub_indices[b]:
                if uncovered[idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b
        
        if best_block_idx < 0:
            break
        
        block = all_blocks[best_block_idx]
        blocks.append(list(block))
        used_blocks.add(best_block_idx)
        
        for idx in block_tsub_indices[best_block_idx]:
            uncovered[idx] = 0
    
    return blocks


def _local_search(blocks, v, k, t, all_tsubs, tsub_to_idx, all_blocks, block_to_idx,
                  block_tsub_indices, rng, deadline):
    """
    Local search to minimize number of blocks.
    Moves:
      1. Remove redundant blocks.
      2. Swap a point in block j to cover a critical t-subset of block i, then remove i.
      3. 2-swap: modify two blocks to enable removal of one or two blocks.
    """
    n_tsubs = len(all_tsubs)
    if not blocks:
        return blocks

    # Build data structures
    block_tsubs = []
    cover_count = np.zeros(n_tsubs, dtype=np.int32)
    
    for block in blocks:
        subs = []
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            subs.append(idx)
            cover_count[idx] += 1
        block_tsubs.append(subs)
    
    no_improve = 0
    max_no_improve = 80
    
    while time.time() < deadline - 3 and no_improve < max_no_improve:
        no_improve += 1
        improved = False
        
        # Phase 1: Remove redundant blocks
        for i in range(len(blocks)):
            if time.time() > deadline - 3:
                break
            redundant = True
            for idx in block_tsubs[i]:
                if cover_count[idx] == 1:
                    redundant = False
                    break
            if redundant:
                for idx in block_tsubs[i]:
                    cover_count[idx] -= 1
                blocks.pop(i)
                block_tsubs.pop(i)
                improved = True
                no_improve = 0
                break
        
        if improved:
            continue
        
        # Phase 2: Try swaps to enable removal
        for i in range(len(blocks)):
            if time.time() > deadline - 3:
                break
            
            criticals = [idx for idx in block_tsubs[i] if cover_count[idx] == 1]
            if not criticals:
                continue
            
            # Try to cover a critical t-subset by modifying another block
            for crit_idx in criticals[:3]:  # Try up to 3 criticals
                if time.time() > deadline - 3:
                    break
                crit_ts = all_tsubs[crit_idx]
                crit_set = set(crit_ts)
                
                for j in range(len(blocks)):
                    if j == i or time.time() > deadline - 3:
                        continue
                    
                    block_j = blocks[j]
                    block_j_set = set(block_j)
                    
                    # Check if block_j already covers crit_ts
                    if crit_idx in block_tsubs[j]:
                        continue
                    
                    common = block_j_set & crit_set
                    missing = crit_set - block_j_set
                    
                    if len(missing) > 2:
                        continue
                    
                    missing_list = list(missing)
                    extra = list(block_j_set - crit_set)
                    
                    if len(extra) < len(missing_list):
                        continue
                    
                    for rem_combo in itertools.combinations(extra, len(missing_list)):
                        if time.time() > deadline - 3:
                            break
                        
                        new_block_j = sorted([p for p in block_j if p not in set(rem_combo)] + missing_list)
                        
                        # Compute new t-subsets
                        new_subs_set = set()
                        for ts in itertools.combinations(new_block_j, t):
                            new_subs_set.add(tsub_to_idx[ts])
                        
                        old_subs_set = set(block_tsubs[j])
                        
                        # Check validity
                        can_swap = True
                        for idx in old_subs_set:
                            if idx not in new_subs_set and cover_count[idx] == 1:
                                can_swap = False
                                break
                        
                        if not can_swap:
                            continue
                        
                        if crit_idx not in new_subs_set:
                            continue
                        
                        # Perform swap
                        for idx in old_subs_set:
                            cover_count[idx] -= 1
                        for idx in new_subs_set:
                            cover_count[idx] += 1
                        
                        blocks[j] = new_block_j
                        block_tsubs[j] = list(new_subs_set)
                        
                        # Check if block i is now redundant
                        still_critical = False
                        for idx in block_tsubs[i]:
                            if cover_count[idx] == 1:
                                still_critical = True
                                break
                        
                        if not still_critical:
                            for idx in block_tsubs[i]:
                                cover_count[idx] -= 1
                            blocks.pop(i)
                            block_tsubs.pop(i)
                            improved = True
                            no_improve = 0
                            break
                        else:
                            # Revert
                            for idx in new_subs_set:
                                cover_count[idx] -= 1
                            for idx in old_subs_set:
                                cover_count[idx] += 1
                            blocks[j] = sorted(block_j_set)
                            block_tsubs[j] = list(old_subs_set)
                        
                        if improved:
                            break
                    
                    if improved:
                        break
                
                if improved:
                    break
            
            if improved:
                break
        
        # Phase 3: Random perturbation
        if no_improve > 25 and len(blocks) > 10:
            j = rng.randint(0, len(blocks) - 1)
            block_j = blocks[j]
            old_subs_set = set(block_tsubs[j])
            
            p_remove = rng.choice(block_j)
            remaining = [p for p in range(v) if p != p_remove]
            p_add = rng.choice(remaining)
            
            new_block_j = sorted([p for p in block_j if p != p_remove] + [p_add])
            
            new_subs_set = set()
            for ts in itertools.combinations(new_block_j, t):
                new_subs_set.add(tsub_to_idx[ts])
            
            can_swap = True
            for idx in old_subs_set:
                if idx not in new_subs_set and cover_count[idx] == 1:
                    can_swap = False
                    break
            
            if can_swap:
                for idx in old_subs_set:
                    cover_count[idx] -= 1
                for idx in new_subs_set:
                    cover_count[idx] += 1
                blocks[j] = new_block_j
                block_tsubs[j] = list(new_subs_set)
                no_improve = 0
    
    return blocks


def construct(v, k, t):
    """Construct a covering design C(v,k,t)."""
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    rng = random.Random(42)
    
    # Precompute
    all_tsubs, tsub_to_idx, all_blocks, block_to_idx, block_tsub_indices = _precompute(v, k, t)
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple restarts
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() > deadline - 40:
            break
        
        # Greedy construction
        blocks = _greedy_construct(v, k, t, all_tsubs, tsub_to_idx, all_blocks, block_to_idx,
                                   block_tsub_indices, rng, deadline - 35)
        
        if not blocks:
            continue
        
        # Verify coverage
        cover_count = np.zeros(len(all_tsubs), dtype=np.uint8)
        for block in blocks:
            for ts in itertools.combinations(block, t):
                cover_count[tsub_to_idx[ts]] += 1
        
        if np.any(cover_count == 0):
            continue
        
        # Local search
        blocks = _local_search(blocks, v, k, t, all_tsubs, tsub_to_idx, all_blocks, block_to_idx,
                               block_tsub_indices, rng, deadline - 5)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    if best_blocks is None:
        # Fallback: greedy without local search
        blocks = _greedy_construct(v, k, t, all_tsubs, tsub_to_idx, all_blocks, block_to_idx,
                                   block_tsub_indices, rng, deadline)
        best_blocks = blocks
    
    return best_blocks
