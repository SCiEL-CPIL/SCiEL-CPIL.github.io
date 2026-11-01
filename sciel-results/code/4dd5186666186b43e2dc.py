"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys
import numpy as np

_global_deadline = None


def _greedy_cover(v, k, t, rng=None, deadline=None):
    """Greedy set-cover: open a block on the first uncovered t-subset, then add
    the point that covers the most uncovered t-subsets until the block has k points."""
    if rng is None:
        rng = random.Random()
    if deadline is None:
        deadline = _global_deadline if _global_deadline else float('inf')
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered and time.time() < deadline:
        # Pick a random uncovered t-subset as seed (randomization helps)
        seed = rng.choice(list(uncovered))
        block = list(seed)
        block_set = set(block)
        while len(block) < k:
            # Compute gain for each candidate point
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Count how many uncovered t-subsets are completed by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    if ts in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x < 0:
                break
            block.append(best_x)
            block_set.add(best_x)
        block.sort()
        blocks.append(block)
        # Remove all t-subsets covered by this block
        block_set = set(block)
        for ts in itertools.combinations(block, t):
            uncovered.discard(ts)
    return blocks


def _build_coverage(blocks, v, k, t):
    """Build coverage count dict: t-subset tuple -> number of blocks covering it."""
    coverage_count = {}
    for block in blocks:
        for ts in itertools.combinations(block, t):
            ts = tuple(sorted(ts))
            coverage_count[ts] = coverage_count.get(ts, 0) + 1
    return coverage_count


def _remove_redundant(blocks, v, k, t, deadline):
    """Iteratively remove redundant blocks (those with no uniquely-covered t-subset)."""
    blocks = [list(b) for b in blocks]
    coverage_count = _build_coverage(blocks, v, k, t)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for bi in range(len(blocks)):
            if time.time() >= deadline:
                break
            redundant = True
            for ts in itertools.combinations(blocks[bi], t):
                ts = tuple(sorted(ts))
                if coverage_count.get(ts, 0) <= 1:
                    redundant = False
                    break
            if redundant:
                removed_block = blocks.pop(bi)
                for ts in itertools.combinations(removed_block, t):
                    ts = tuple(sorted(ts))
                    coverage_count[ts] -= 1
                    if coverage_count[ts] <= 0:
                        del coverage_count[ts]
                improved = True
                break
    return blocks, coverage_count


def _maxsat_local_search(blocks, v, k, t, deadline):
    """MaxSAT-style local search: try to reduce block count by swapping out
    blocks that are 'least useful' (cover fewest critical t-subsets) with
    blocks that cover more critical subsets. Also try direct removal."""
    blocks = [list(b) for b in blocks]
    coverage_count = _build_coverage(blocks, v, k, t)
    rng = random.Random(123)

    n_blocks = len(blocks)
    # Precompute for each block: number of t-subsets it uniquely covers (count==1)
    # and total t-subsets it covers
    block_stats = []
    for block in blocks:
        unique = 0
        total = 0
        for ts in itertools.combinations(block, t):
            ts = tuple(sorted(ts))
            total += 1
            if coverage_count.get(ts, 0) == 1:
                unique += 1
        block_stats.append((unique, total))

    improved = True
    while improved and time.time() < deadline:
        improved = False

        # First pass: try to remove blocks with 0 unique t-subsets
        for bi in range(len(blocks)):
            if time.time() >= deadline:
                break
            if block_stats[bi][0] == 0:
                # Block is redundant
                removed_block = blocks.pop(bi)
                block_stats.pop(bi)
                for ts in itertools.combinations(removed_block, t):
                    ts = tuple(sorted(ts))
                    coverage_count[ts] -= 1
                    if coverage_count[ts] <= 0:
                        del coverage_count[ts]
                improved = True
                break
        if improved:
            continue

        # Second pass: try swap - remove a block with few unique covers,
        # and see if we can cover all its unique t-subsets by modifying another block
        # or by adding a new block that covers them (but that doesn't reduce count)
        # Better: try to find a block that, when removed, its unique t-subsets
        # are all covered by other blocks after we slightly modify one block.

        # Simpler approach: try removing a block and then greedily patching
        # uncovered t-subsets by modifying existing blocks (adding points)
        # or adding minimal new blocks. If total count decreases, keep it.

        # Try each block as a removal candidate (prioritize low unique count)
        candidates = sorted(range(len(blocks)), key=lambda i: block_stats[i][0])
        tried = 0
        for bi in candidates:
            if time.time() >= deadline or tried >= 5:
                break
            tried += 1
            if block_stats[bi][0] > 3:
                break  # Too many unique subsets, skip

            test_blocks = [list(b) for b in blocks]
            test_cov = dict(coverage_count)
            removed = test_blocks.pop(bi)
            for ts in itertools.combinations(removed, t):
                ts = tuple(sorted(ts))
                test_cov[ts] -= 1
                if test_cov[ts] <= 0:
                    del test_cov[ts]

            # Find uncovered t-subsets
            uncovered = []
            for ts in itertools.combinations(removed, t):
                ts = tuple(sorted(ts))
                if test_cov.get(ts, 0) == 0:
                    uncovered.append(ts)

            if not uncovered:
                # Shouldn't happen since unique > 0 check, but safety
                continue

            # Try to cover uncovered t-subsets by modifying existing blocks
            # Strategy: for each uncovered t-subset, find a block that already
            # contains t-1 of its points, and add the missing point
            success = True
            for ts in uncovered:
                ts_set = set(ts)
                found = False
                for j, block in enumerate(test_blocks):
                    if time.time() >= deadline:
                        success = False
                        break
                    block_set = set(block)
                    # Check if block contains at least t-1 points of ts
                    overlap = len(ts_set & block_set)
                    if overlap >= t - 1:
                        missing = ts_set - block_set
                        if len(missing) == 1 and len(block) < k:
                            # Add the missing point
                            new_point = missing.pop()
                            test_blocks[j].append(new_point)
                            test_blocks[j].sort()
                            # Update coverage
                            for new_ts in itertools.combinations(test_blocks[j], t):
                                new_ts = tuple(sorted(new_ts))
                                test_cov[new_ts] = test_cov.get(new_ts, 0) + 1
                            found = True
                            break
                if not found:
                    # Try adding a new block containing this t-subset
                    # But this increases count, so only do it if we can
                    # later remove more blocks. For now, mark as failure.
                    success = False
                    break

            if success and len(test_blocks) < len(blocks):
                # Accept the swap
                blocks = test_blocks
                coverage_count = test_cov
                block_stats = []
                for block in blocks:
                    unique = 0
                    total = 0
                    for ts in itertools.combinations(block, t):
                        ts = tuple(sorted(ts))
                        total += 1
                        if coverage_count.get(ts, 0) == 1:
                            unique += 1
                    block_stats.append((unique, total))
                improved = True
                break

        # Third pass: try to merge/replace two blocks with one
        # Find two blocks that together cover a set of t-subsets,
        # and see if a single block can cover all of them
        if not improved and time.time() < deadline:
            n = len(blocks)
            tried_pairs = 0
            for i in range(min(n, 50)):
                if time.time() >= deadline or tried_pairs >= 20:
                    break
                for j in range(i + 1, min(n, 50)):
                    if time.time() >= deadline or tried_pairs >= 20:
                        break
                    tried_pairs += 1
                    # Get all t-subsets covered by block i or block j
                    # Check if there exists a k-subset that covers all of them
                    # This is expensive, so only try for small overlap
                    set_i = set(blocks[i])
                    set_j = set(blocks[j])
                    union_size = len(set_i | set_j)
                    if union_size > k:
                        continue
                    # The union fits in a single block
                    # Check if union block covers all t-subsets that i and j cover
                    union_block = sorted(set_i | set_j)
                    if len(union_block) < k:
                        # Pad with random points
                        remaining = set(range(v)) - set(union_block)
                        extra = rng.sample(sorted(remaining), k - len(union_block))
                        union_block = sorted(union_block + extra)

                    # Check coverage
                    test_blocks = [list(b) for b in blocks if b != blocks[i] and b != blocks[j]]
                    test_blocks.append(union_block)
                    test_cov = _build_coverage(test_blocks, v, k, t)

                    # Verify all t-subsets still covered
                    valid = True
                    all_ts = set(itertools.combinations(range(v), t))
                    # Only need to check t-subsets that were covered by i or j
                    for ts in itertools.combinations(blocks[i], t):
                        ts = tuple(sorted(ts))
                        if test_cov.get(ts, 0) == 0:
                            valid = False
                            break
                    if valid:
                        for ts in itertools.combinations(blocks[j], t):
                            ts = tuple(sorted(ts))
                            if test_cov.get(ts, 0) == 0:
                                valid = False
                                break
                    if valid and len(test_blocks) < len(blocks):
                        blocks = test_blocks
                        coverage_count = test_cov
                        block_stats = []
                        for block in blocks:
                            unique = 0
                            total = 0
                            for ts in itertools.combinations(block, t):
                                ts = tuple(sorted(ts))
                                total += 1
                                if coverage_count.get(ts, 0) == 1:
                                    unique += 1
                            block_stats.append((unique, total))
                        improved = True
                        break
                if improved:
                    break

    return blocks


def _block_replacement_search(blocks, v, k, t, deadline):
    """Try to replace existing blocks with better ones that cover more critical t-subsets.
    This is a Max-SAT inspired approach where we treat each block as a variable and
    try to find replacements that improve coverage of critical (low-count) t-subsets."""
    blocks = [list(b) for b in blocks]
    coverage_count = _build_coverage(blocks, v, k, t)
    rng = random.Random(42)
    
    # Precompute critical t-subsets (those with low coverage)
    critical_ts = {}
    for ts, cnt in coverage_count.items():
        if cnt <= 2:
            critical_ts[ts] = cnt
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # For each block, try to find a replacement that covers more critical t-subsets
        for bi in range(len(blocks)):
            if time.time() >= deadline:
                break
            
            # Compute how many critical t-subsets this block covers
            block_critical = set()
            for ts in itertools.combinations(blocks[bi], t):
                ts = tuple(sorted(ts))
                if ts in critical_ts:
                    block_critical.add(ts)
            
            # Try random replacements
            for attempt in range(20):
                if time.time() >= deadline:
                    break
                
                # Create a random replacement block that includes some points from original
                # and some random points
                orig_set = set(blocks[bi])
                new_block = list(rng.sample(sorted(orig_set), min(3, len(orig_set))))
                
                # Add random points to fill up to k
                remaining = [x for x in range(v) if x not in new_block]
                new_block.extend(rng.sample(remaining, k - len(new_block)))
                new_block.sort()
                
                if new_block == blocks[bi]:
                    continue
                
                # Compute how many critical t-subsets the new block covers
                new_critical = set()
                for ts in itertools.combinations(new_block, t):
                    ts = tuple(sorted(ts))
                    if ts in critical_ts:
                        new_critical.add(ts)
                
                # Accept if new block covers more critical t-subsets
                if len(new_critical) > len(block_critical):
                    # Verify that removing old block and adding new one doesn't break coverage
                    test_blocks = [list(b) for b in blocks if b != blocks[bi]]
                    test_blocks.append(new_block)
                    test_cov = _build_coverage(test_blocks, v, k, t)
                    
                    # Check all t-subsets are still covered
                    valid = True
                    for ts in itertools.combinations(blocks[bi], t):
                        ts = tuple(sorted(ts))
                        if test_cov.get(ts, 0) == 0:
                            valid = False
                            break
                    
                    if valid:
                        blocks[bi] = new_block
                        coverage_count = test_cov
                        # Update critical ts
                        critical_ts = {}
                        for ts, cnt in coverage_count.items():
                            if cnt <= 2:
                                critical_ts[ts] = cnt
                        improved = True
                        break
        
        # Also try to remove blocks that are now redundant
        for bi in range(len(blocks)):
            if time.time() >= deadline:
                break
            redundant = True
            for ts in itertools.combinations(blocks[bi], t):
                ts = tuple(sorted(ts))
                if coverage_count.get(ts, 0) <= 1:
                    redundant = False
                    break
            if redundant:
                removed_block = blocks.pop(bi)
                for ts in itertools.combinations(removed_block, t):
                    ts = tuple(sorted(ts))
                    coverage_count[ts] -= 1
                    if coverage_count[ts] <= 0:
                        del coverage_count[ts]
                improved = True
                break
    
    return blocks


def construct(v, k, t):
    global _global_deadline
    _global_deadline = time.time() + 110  # Safety margin before 120s limit
    rng = random.Random(42)

    # Phase 1: Greedy cover with randomization, multiple restarts
    best_blocks = None
    best_count = float('inf')

    # First greedy run
    blocks = _greedy_cover(v, k, t, rng, _global_deadline)
    best_blocks = blocks
    best_count = len(blocks)

    # Try more greedy restarts if time permits
    restarts = 0
    while time.time() < _global_deadline - 60 and restarts < 5:
        blocks = _greedy_cover(v, k, t, rng, _global_deadline - 60)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        restarts += 1

    # Phase 2: Remove redundant blocks
    if time.time() < _global_deadline - 20:
        best_blocks, _ = _remove_redundant(best_blocks, v, k, t, _global_deadline - 20)

    # Phase 3: Block replacement search (Max-SAT inspired)
    if time.time() < _global_deadline - 10:
        best_blocks = _block_replacement_search(best_blocks, v, k, t, _global_deadline - 5)

    # Phase 4: Final redundancy removal
    if time.time() < _global_deadline - 2:
        best_blocks, _ = _remove_redundant(best_blocks, v, k, t, _global_deadline - 1)

    return best_blocks
# EVOLVE-BLOCK-END
