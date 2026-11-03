"""Improved covering design solver using randomized greedy with local search."""
import time
import random
import itertools
import numpy as np
from collections import defaultdict


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index for fast lookup
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # For each point, precompute which t-subsets it participates in
    point_tsubs = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubsets):
        for p in ts:
            point_tsubs[p].append(i)
    
    # Precompute for each t-subset its bitmask representation (for fast containment checks)
    # We'll use integer bitmasks for blocks
    point_mask = [1 << p for p in range(v)]
    
    # For each block (as a bitmask), precompute which t-subsets it covers
    # tsub_mask[i] = bitmask of points in t-subset i
    tsub_mask = []
    for ts in all_tsubsets:
        m = 0
        for p in ts:
            m |= point_mask[p]
        tsub_mask.append(m)
    
    # Precompute: for each possible (t-1)-subset of points, which t-subsets contain it
    tminus1_to_tsubs = defaultdict(list)
    for i, ts in enumerate(all_tsubsets):
        ts_list = list(ts)
        for j in range(t):
            sub = tuple(sorted(ts_list[:j] + ts_list[j+1:]))
            tminus1_to_tsubs[sub].append(i)
    
    # Precompute for each point x: list of (t_sub_idx, mask_of_other_t_minus_1_points)
    # This allows fast gain computation
    point_tsub_info = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubsets):
        ts_list = list(ts)
        for j in range(t):
            x = ts_list[j]
            others = [p for p in ts_list if p != x]
            mask_others = 0
            for p in others:
                mask_others |= point_mask[p]
            point_tsub_info[x].append((i, mask_others))
    
    # Convert to numpy arrays for faster operations
    # For each point, store as two parallel arrays
    point_tsub_idx = []
    point_tsub_mask_others = []
    for x in range(v):
        idxs = [info[0] for info in point_tsub_info[x]]
        masks = [info[1] for info in point_tsub_info[x]]
        point_tsub_idx.append(np.array(idxs, dtype=np.int32))
        point_tsub_mask_others.append(np.array(masks, dtype=np.int64))
    
    # Precompute: for each t-subset index, which points are in it (as a list)
    # and a bitmask
    tsub_points = [list(ts) for ts in all_tsubsets]
    
    # Precompute for each point: which t-subsets contain it (list of indices)
    # This is point_tsub_idx already, but let's also have a set for fast membership
    # Actually we need: for a block mask, which t-subsets are covered?
    # t-subset i is covered by block B iff tsub_mask[i] & ~block_mask == 0
    # i.e., all points of t-subset i are in B.
    
    # For fast coverage update: precompute for each point p, the list of t-subset indices
    # that contain p. Then a t-subset is covered by block B iff it's in the intersection
    # of all point_tsub_lists for points in B.
    # Alternative: for each t-subset, store its point bitmask. Then check containment.
    # Vectorized: create a 2D array where row i is the bitmask of t-subset i.
    # Then for a block mask bm, covered = (tsub_mask_arr & ~bm) == 0
    
    tsub_mask_arr = np.array(tsub_mask, dtype=np.int64)
    
    # Greedy with randomization
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = np.ones(num_tsubsets, dtype=np.bool_)
        
        blocks = []
        remaining = num_tsubsets
        
        while remaining > 0:
            if time.time() > deadline:
                return None  # timeout
            
            # Pick a random uncovered t-subset to seed the block
            idx = rng.randrange(num_tsubsets)
            while not uncovered[idx]:
                idx = (idx + 1) % num_tsubsets
            
            seed_tsub = all_tsubsets[idx]
            block = list(seed_tsub)
            rng.shuffle(block)
            
            # Greedily add points to maximize gain
            while len(block) < k:
                if time.time() > deadline:
                    return None
                
                best_gain = -1
                best_point = -1
                block_set = set(block)
                block_mask = 0
                for p in block:
                    block_mask |= point_mask[p]
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    # Calculate gain using numpy
                    idxs = point_tsub_idx[x]
                    masks = point_tsub_mask_others[x]
                    
                    # Filter to only uncovered t-subsets
                    mask_uncovered = uncovered[idxs]
                    if not mask_uncovered.any():
                        continue
                    
                    # Check which of the uncovered t-subsets have their other t-1 points in block
                    # mask_others & ~block_mask == 0 means all other points are in block
                    candidates = masks[mask_uncovered]
                    valid = (candidates & ~block_mask) == 0
                    gain = int(valid.sum())
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                    elif gain == best_gain and best_point != -1:
                        # Tie-breaking: random
                        if rng.random() < 0.5:
                            best_point = x
                
                if best_point == -1 or best_gain <= 0:
                    # No useful point, just pick a random one
                    candidates = [x for x in range(v) if x not in block_set]
                    if candidates:
                        best_point = rng.choice(candidates)
                    else:
                        break
                
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered using vectorized operation
            block_mask = 0
            for p in block:
                block_mask |= point_mask[p]
            
            # Vectorized update: find all t-subsets covered by this block
            newly_covered = (tsub_mask_arr & ~block_mask) == 0
            # Only update those that were previously uncovered
            to_update = newly_covered & uncovered
            num_new = int(to_update.sum())
            if num_new > 0:
                uncovered[to_update] = False
                remaining -= num_new
        
        return blocks
    
    def remove_redundant(blocks, time_limit=None):
        """Remove blocks whose t-subsets are all covered by other blocks."""
        if time_limit is None:
            time_limit = deadline
        n = len(blocks)
        if n == 0:
            return blocks
        
        # Build coverage: for each t-subset, count how many blocks cover it
        cover_count = np.zeros(num_tsubsets, dtype=np.int32)
        
        block_masks = []
        for b in blocks:
            m = 0
            for p in b:
                m |= point_mask[p]
            block_masks.append(m)
        
        # Also store which t-subsets each block covers for later use
        block_covers = []
        for bm in block_masks:
            covered_mask = (tsub_mask_arr & ~bm) == 0
            covered = np.where(covered_mask)[0].tolist()
            block_covers.append(covered)
            cover_count[covered] += 1
        
        # A block is redundant if every t-subset it covers is also covered by another block
        keep = [True] * n
        for bi in range(n):
            if not keep[bi]:
                continue
            if time.time() > time_limit:
                break
            redundant = True
            for ts_idx in block_covers[bi]:
                if cover_count[ts_idx] == 1:
                    # Only this block covers it, so not redundant
                    redundant = False
                    break
            if redundant:
                keep[bi] = False
                # Decrement cover counts for t-subsets covered by this block
                for ts_idx in block_covers[bi]:
                    cover_count[ts_idx] -= 1
        
        if all(keep):
            return blocks
        
        new_blocks = [b for i, b in enumerate(blocks) if keep[i]]
        return new_blocks
    
    def local_search(blocks, seed, time_limit=None):
        """Try to remove blocks one at a time if the covering is still valid."""
        if time_limit is None:
            time_limit = deadline
        rng = random.Random(seed)
        n = len(blocks)
        
        # Count how many blocks cover each t-subset
        cover_count = np.zeros(num_tsubsets, dtype=np.int32)
        block_masks = []
        for b in blocks:
            m = 0
            for p in b:
                m |= point_mask[p]
            block_masks.append(m)
        
        # Precompute which t-subsets each block covers using vectorized ops
        block_covers = []
        for bm in block_masks:
            covered_mask = (tsub_mask_arr & ~bm) == 0
            covered = np.where(covered_mask)[0].tolist()
            block_covers.append(covered)
            cover_count[covered] += 1
        
        # Try removing blocks in random order
        order = list(range(n))
        rng.shuffle(order)
        
        kept = [True] * n
        for bi in order:
            if time.time() > time_limit:
                break
            if not kept[bi]:
                continue
            # Check if removing this block leaves any t-subset uncovered
            can_remove = True
            for ts_idx in block_covers[bi]:
                if cover_count[ts_idx] == 1:
                    can_remove = False
                    break
            if can_remove:
                kept[bi] = False
                for ts_idx in block_covers[bi]:
                    cover_count[ts_idx] -= 1
        
        new_blocks = [b for i, b in enumerate(blocks) if kept[i]]
        
        # Second pass: try swapping - replace a block with a smaller/different one
        # For each block, check if we can find another k-subset that covers all
        # t-subsets uniquely covered by this block, using fewer or different blocks.
        # This is expensive, so we only do a limited version.
        
        # Rebuild cover counts for the kept blocks
        n2 = len(new_blocks)
        if n2 < n and n2 > 0:
            cover_count2 = np.zeros(num_tsubsets, dtype=np.int32)
            block_masks2 = []
            for b in new_blocks:
                m = 0
                for p in b:
                    m |= point_mask[p]
                block_masks2.append(m)
            
            block_covers2 = []
            for bm in block_masks2:
                covered_mask = (tsub_mask_arr & ~bm) == 0
                covered = np.where(covered_mask)[0].tolist()
                block_covers2.append(covered)
                cover_count2[covered] += 1
            
            # Try to remove more blocks
            order2 = list(range(n2))
            rng.shuffle(order2)
            kept2 = [True] * n2
            for bi in order2:
                if time.time() > time_limit:
                    break
                if not kept2[bi]:
                    continue
                can_remove = True
                for ts_idx in block_covers2[bi]:
                    if cover_count2[ts_idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    kept2[bi] = False
                    for ts_idx in block_covers2[bi]:
                        cover_count2[ts_idx] -= 1
            
            if sum(kept2) < n2:
                new_blocks = [b for i, b in enumerate(new_blocks) if kept2[i]]
        
        return new_blocks
    
    # Try multiple random seeds and keep the best
    best_blocks = None
    best_count = float('inf')
    
    # First, try a deterministic greedy (seed 0)
    blocks = greedy_cover(0)
    if blocks is not None:
        # Apply redundancy removal
        blocks = remove_redundant(blocks)
        # Apply local search
        blocks = local_search(blocks, 0)
        best_blocks = blocks
        best_count = len(blocks)
    
    # Try more seeds with time remaining
    seed = 1
    while time.time() < deadline - 15:
        blocks = greedy_cover(seed)
        if blocks is not None:
            # Apply redundancy removal
            blocks = remove_redundant(blocks, time_limit=deadline - 10)
            # Apply local search
            blocks = local_search(blocks, seed, time_limit=deadline - 5)
            if len(blocks) < best_count:
                best_blocks = blocks
                best_count = len(blocks)
        seed += 1
    
    # Final polish: if we have time, try local search on the best solution again
    if best_blocks is not None and time.time() < deadline - 3:
        polished = local_search(best_blocks, 999, time_limit=deadline - 1)
        if len(polished) < best_count:
            best_blocks = polished
            best_count = len(best_blocks)
    
    if best_blocks is None:
        # Fallback: return empty (shouldn't happen)
        return []
    
    return best_blocks
