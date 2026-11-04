"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(v, k, t):
    deadline = time.time() + 140.0  # safety margin before 150s kill
    rng = random.Random(42)
    
    # Precompute all t-subsets and index them
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # For each point x, precompute which t-subsets contain x
    # point_tsubs[x] = list of indices of t-subsets containing x
    point_tsubs = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubs):
        for x in ts:
            point_tsubs[x].append(i)
    
    # Greedy covering with bitset tracking
    # uncovered_mask: bit i set means t-subset i is uncovered
    uncovered_mask = (1 << n_tsubs) - 1
    blocks = []
    
    while uncovered_mask:
        if time.time() > deadline - 5:
            break
        
        # Find an uncovered t-subset to start a new block
        # Pick the one with fewest possible extensions (most constrained)
        best_start = None
        best_count = float('inf')
        
        # Sample some uncovered t-subsets to find a good starting point
        sample_size = min(50, n_tsubs)
        while True:
            # Get a random uncovered t-subset
            bits = uncovered_mask
            while bits:
                idx = (bits & -bits).bit_length() - 1
                ts = all_tsubs[idx]
                
                # Count how many points can be added to cover this t-subset
                # (all points not in ts)
                count = v - t
                if count < best_count:
                    best_count = count
                    best_start = ts
                bits &= bits - 1
                if bits == 0:
                    break
            
            if best_count <= t + 1:
                break
            if sample_size <= 0:
                break
            sample_size -= 1
        
        if best_start is None:
            # Fallback: just pick any uncovered t-subset
            bits = uncovered_mask
            idx = (bits & -bits).bit_length() - 1
            best_start = all_tsubs[idx]
        
        # Build the block greedily
        block = list(best_start)
        block_set = set(block)
        
        while len(block) < k:
            if time.time() > deadline - 5:
                break
            
            # Find the point that covers the most uncovered t-subsets
            best_gain = -1
            best_point = None
            candidates = [x for x in range(v) if x not in block_set]
            
            # Shuffle candidates for tie-breaking variety
            rng.shuffle(candidates)
            
            for x in candidates:
                # Count uncovered t-subsets that contain x and all but one element of block
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    # Check if sorted(rest + (x,)) is an uncovered t-subset
                    ts = tuple(sorted(rest + (x,)))
                    idx = tsub_to_idx.get(ts)
                    if idx is not None and (uncovered_mask >> idx) & 1:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point is None:
                break
            block.append(best_point)
            block_set.add(best_point)
        
        if len(block) < k:
            # Couldn't complete the block, skip it
            continue
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered_mask: remove all t-subsets contained in this block
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            uncovered_mask &= ~(1 << idx)
    
    # If we ran out of time, we may have uncovered t-subsets.
    # Try to cover them with additional blocks quickly.
    while uncovered_mask and time.time() < deadline - 2:
        bits = uncovered_mask
        idx = (bits & -bits).bit_length() - 1
        ts = all_tsubs[idx]
        block = list(ts)
        block_set = set(block)
        
        while len(block) < k:
            best_gain = -1
            best_point = None
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    t_new = tuple(sorted(rest + (x,)))
                    i = tsub_to_idx.get(t_new)
                    if i is not None and (uncovered_mask >> i) & 1:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point is None:
                break
            block.append(best_point)
            block_set.add(best_point)
        
        if len(block) == k:
            block.sort()
            blocks.append(block)
            for ts2 in itertools.combinations(block, t):
                i = tsub_to_idx[ts2]
                uncovered_mask &= ~(1 << i)
        else:
            break
    
    # Local search: try to replace pairs of blocks with a single block
    if time.time() < deadline - 10:
        blocks = _local_search(blocks, v, k, t, tsub_to_idx, all_tsubs, deadline, rng)
    
    # Final validation pass - add blocks for any remaining uncovered t-subsets
    if time.time() < deadline - 2:
        # Rebuild coverage from scratch
        covered = set()
        for b in blocks:
            for ts in itertools.combinations(b, t):
                covered.add(ts)
        
        remaining = set(all_tsubs) - covered
        for ts in remaining:
            if time.time() > deadline - 2:
                break
            block = list(ts)
            block_set = set(block)
            while len(block) < k:
                best_gain = -1
                best_point = None
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        t_new = tuple(sorted(rest + (x,)))
                        if t_new in remaining:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point is None:
                    break
                block.append(best_point)
                block_set.add(best_point)
            if len(block) == k:
                block.sort()
                blocks.append(block)
                for ts2 in itertools.combinations(block, t):
                    remaining.discard(ts2)
    
    return blocks


def _local_search(blocks, v, k, t, tsub_to_idx, all_tsubs, deadline, rng):
    """Try to reduce block count by merging/eliminating redundant blocks."""
    if not blocks:
        return blocks
    
    # Compute coverage for each block
    block_tsubs = []
    for b in blocks:
        tsubs = set()
        for ts in itertools.combinations(b, t):
            tsubs.add(ts)
        block_tsubs.append(tsubs)
    
    # Try removing each block and checking if coverage is still complete
    all_covered = set()
    for ts_set in block_tsubs:
        all_covered.update(ts_set)
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        n = len(blocks)
        # Try removing blocks in random order
        indices = list(range(n))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 10:
                break
            # Check if block i's t-subsets are all covered by other blocks
            redundant = True
            for ts in block_tsubs[i]:
                covered_by_other = False
                for j in range(n):
                    if j == i:
                        continue
                    if ts in block_tsubs[j]:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    redundant = False
                    break
            
            if redundant:
                del blocks[i]
                del block_tsubs[i]
                improved = True
    
    return blocks
# EVOLVE-BLOCK-END
