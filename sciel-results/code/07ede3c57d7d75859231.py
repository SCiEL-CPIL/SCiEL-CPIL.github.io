import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute all k-blocks
    all_kblocks = list(itertools.combinations(range(v), k))
    n_kblocks = len(all_kblocks)
    
    # For each k-block, compute which t-subsets it covers
    # Use a list of lists of t-subset indices
    block_covers = [None] * n_kblocks
    for i, block in enumerate(all_kblocks):
        cov = []
        for ts in itertools.combinations(block, t):
            cov.append(tsubset_to_idx[ts])
        block_covers[i] = cov
    
    # Initialize: greedy set cover
    uncovered_count = np.zeros(n_tsubsets, dtype=np.int32)
    uncovered_count[:] = 1
    blocks = []
    used_blocks = set()
    
    # Greedy phase
    while (time.time() - start_time) < deadline - 20:
        best_idx = -1
        best_gain = 0
        for i, block in enumerate(all_kblocks):
            if i in used_blocks:
                continue
            gain = 0
            for ts_idx in block_covers[i]:
                if uncovered_count[ts_idx] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = i
            if gain == 0:
                continue
        if best_idx == -1 or best_gain == 0:
            break
        blocks.append(best_idx)
        used_blocks.add(best_idx)
        for ts_idx in block_covers[best_idx]:
            uncovered_count[ts_idx] = 0
    
    # Check if we've covered everything
    if np.all(uncovered_count == 0):
        return [list(all_kblocks[i]) for i in blocks]
    
    # Local search: try to improve by swapping points in blocks
    # Represent blocks as sets for faster operations
    block_sets = [set(all_kblocks[i]) for i in blocks]
    
    # Build coverage map: for each t-subset, which blocks cover it
    tsub_covered_by = defaultdict(set)
    for b_idx, block_set in enumerate(block_sets):
        for ts in itertools.combinations(block_set, t):
            tsub_covered_by[ts].add(b_idx)
    
    def is_covered(ts):
        return len(tsub_covered_by[ts]) > 0
    
    # Try to reduce number of blocks by removing redundant ones
    changed = True
    while changed and (time.time() - start_time) < deadline - 10:
        changed = False
        for i in range(len(block_sets) - 1, -1, -1):
            # Check if block i is redundant
            redundant = True
            for ts in itertools.combinations(block_sets[i], t):
                if len(tsub_covered_by[ts]) == 1:
                    redundant = False
                    break
            if redundant:
                # Remove block i
                for ts in itertools.combinations(block_sets[i], t):
                    tsub_covered_by[ts].discard(i)
                block_sets.pop(i)
                blocks.pop(i)
                # Re-index tsub_covered_by
                for ts, coverers in tsub_covered_by.items():
                    new_coverers = set()
                    for c in coverers:
                        if c < i:
                            new_coverers.add(c)
                        elif c > i:
                            new_coverers.add(c - 1)
                    tsub_covered_by[ts] = new_coverers
                changed = True
                break
    
    # Local search: try swapping points in blocks to increase coverage
    rng = random.Random(42)
    
    def get_uncovered_count():
        count = 0
        for ts in all_tsubsets:
            if len(tsub_covered_by[ts]) == 0:
                count += 1
        return count
    
    current_uncovered = get_uncovered_count()
    
    # Try to swap points in blocks
    for iteration in range(1000):
        if (time.time() - start_time) > deadline - 5:
            break
        if current_uncovered == 0:
            break
        
        # Pick a random block and try to swap a point
        b_idx = rng.randint(0, len(block_sets) - 1)
        block = list(block_sets[b_idx])
        point_in_block = rng.choice(block)
        
        # Try swapping with a random point not in the block
        other_points = [p for p in range(v) if p not in block_sets[b_idx]]
        if not other_points:
            continue
        new_point = rng.choice(other_points)
        
        # Compute change in coverage
        old_block = block_sets[b_idx]
        new_block = old_block - {point_in_block} | {new_point}
        
        # Count how many t-subsets are newly covered and how many are lost
        newly_covered = 0
        lost = 0
        for ts in itertools.combinations(old_block, t):
            if new_point not in ts and point_in_block in ts:
                if len(tsub_covered_by[ts]) == 1:
                    lost += 1
        for ts in itertools.combinations(new_block, t):
            if point_in_block not in ts and new_point in ts:
                if len(tsub_covered_by[ts]) == 0:
                    newly_covered += 1
        
        if newly_covered > lost:
            # Accept the swap
            for ts in itertools.combinations(old_block, t):
                if point_in_block in ts and new_point not in ts:
                    tsub_covered_by[ts].discard(b_idx)
                    if len(tsub_covered_by[ts]) == 0:
                        pass
            for ts in itertools.combinations(new_block, t):
                if new_point in ts and point_in_block not in ts:
                    tsub_covered_by[ts].add(b_idx)
            block_sets[b_idx] = new_block
            current_uncovered = get_uncovered_count()
    
    # Final check
    final_uncovered = get_uncovered_count()
    if final_uncovered == 0:
        return [list(block_sets[i]) for i in range(len(block_sets))]
    
    # If not fully covered, use greedy to fill in
    uncovered_set = set()
    for ts in all_tsubsets:
        if len(tsub_covered_by[ts]) == 0:
            uncovered_set.add(ts)
    
    if uncovered_set:
        # Add blocks to cover remaining
        while uncovered_set and (time.time() - start_time) < deadline - 2:
            best_idx = -1
            best_gain = 0
            for i, block in enumerate(all_kblocks):
                if i in used_blocks:
                    continue
                gain = 0
                for ts in itertools.combinations(block, t):
                    if ts in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1 or best_gain == 0:
                break
            blocks.append(best_idx)
            used_blocks.add(best_idx)
            block_sets.append(set(all_kblocks[best_idx]))
            for ts in itertools.combinations(all_kblocks[best_idx], t):
                if ts in uncovered_set:
                    uncovered_set.discard(ts)
    
    return [list(block_sets[i]) for i in range(len(block_sets))]
