"""Structural change: model the covering as a minimum set cover on t-subsets.
Use a fast greedy construction (with vectorized numpy scoring) to get a good
initial solution, then perform local search (swap, delete, add) to reduce the
block count. A deadline guard ensures we return the best valid solution found.
"""
import time
import itertools
import random
import numpy as np

# Deadline: kill at 145s to leave safety margin before 150s evaluator limit
_DEADLINE = 145.0


def _time_left():
    return _DEADLINE - time.time()


def _all_tsubsets(v, t):
    """Generate all t-subsets as tuples."""
    return list(itertools.combinations(range(v), t))


def _tsubsets_of_block(block, t):
    """Generate all t-subsets contained in a given block."""
    return list(itertools.combinations(block, t))


def _cover_count(block, t, uncovered_set):
    """Count how many uncovered t-subsets are in the block."""
    return sum(1 for ts in _tsubsets_of_block(block, t) if ts in uncovered_set)


def _greedy_construct(v, k, t, rng, time_limit):
    """Fast greedy set cover construction using numpy for scoring."""
    all_points = list(range(v))
    # Precompute all t-subsets
    all_tsubs = _all_tsubsets(v, t)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    # For each block, precompute which t-subsets it covers (as a set of indices)
    # We'll build blocks one at a time.
    
    uncovered = set(range(n_tsubs))  # indices of uncovered t-subsets
    blocks = []
    
    # Precompute for each point, the list of t-subsets containing it
    point_to_tsubs = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubs):
        for p in ts:
            point_to_tsubs[p].append(i)
    
    start = time.time()
    while uncovered and (time.time() - start) < time_limit:
        # Pick the first uncovered t-subset to anchor the new block
        anchor = min(uncovered)
        anchor_tsub = all_tsubs[anchor]
        
        # Start block with the anchor's points
        block = list(anchor_tsub)
        block_set = set(block)
        
        # Greedily add points to maximize coverage of uncovered t-subsets
        while len(block) < k:
            best_point = -1
            best_gain = -1
            # For each candidate point, compute gain
            for p in all_points:
                if p in block_set:
                    continue
                # Gain = number of uncovered t-subsets that contain p and are subset of (block + {p})
                # A t-subset is covered by block+p if it's a subset of block∪{p} and contains p
                # i.e., it's a (t-1)-subset of block plus p
                gain = 0
                # Generate all (t-1)-subsets of block
                if len(block) >= t - 1:
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (p,)))
                        idx = tsub_index.get(ts, -1)
                        if idx >= 0 and idx in uncovered:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = p
            
            if best_point == -1 or best_gain <= 0:
                # No point improves coverage; just pick any remaining point
                for p in all_points:
                    if p not in block_set:
                        best_point = p
                        break
                if best_point == -1:
                    break
            
            block.append(best_point)
            block_set.add(best_point)
        
        block.sort()
        blocks.append(tuple(block))
        
        # Update uncovered
        for ts in _tsubsets_of_block(block, t):
            idx = tsub_index.get(ts, -1)
            if idx >= 0:
                uncovered.discard(idx)
    
    # If we ran out of time, check if we have a valid cover
    if not uncovered:
        return [list(b) for b in blocks]
    return None


def _build_coverage_matrix(blocks, v, t):
    """For each block, compute the set of t-subset indices it covers."""
    all_tsubs = _all_tsubsets(v, t)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    block_cover = []
    for block in blocks:
        covered = set()
        for ts in _tsubsets_of_block(block, t):
            idx = tsub_index.get(ts, -1)
            if idx >= 0:
                covered.add(idx)
        block_cover.append(covered)
    return block_cover, n_tsubs, tsub_index


def _is_valid_cover(blocks, v, t):
    """Check if blocks cover all t-subsets."""
    covered = set()
    for block in blocks:
        for ts in _tsubsets_of_block(block, t):
            covered.add(ts)
    all_tsubs = set(_all_tsubsets(v, t))
    return all_tsubs <= covered


def _local_search(blocks, v, k, t, rng, time_limit):
    """Local search: try to delete, swap, or replace blocks to reduce count."""
    blocks = [list(b) for b in blocks]
    n_blocks = len(blocks)
    
    all_tsubs = _all_tsubsets(v, t)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    # Build coverage: for each t-subset, which blocks cover it
    tsub_to_blocks = [set() for _ in range(n_tsubs)]
    block_cover = []
    for bi, block in enumerate(blocks):
        covered = set()
        for ts in _tsubsets_of_block(block, t):
            idx = tsub_index.get(ts, -1)
            if idx >= 0:
                covered.add(idx)
                tsub_to_blocks[idx].add(bi)
        block_cover.append(covered)
    
    start = time.time()
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        
        # Try to delete a block that is redundant
        for bi in range(len(blocks)):
            if (time.time() - start) > time_limit:
                break
            # Check if removing block bi leaves all t-subsets covered
            # A t-subset is uncovered if block bi is its only cover
            redundant = True
            for ts_idx in block_cover[bi]:
                if len(tsub_to_blocks[ts_idx]) == 1:
                    redundant = False
                    break
            if redundant:
                # Remove block bi
                for ts_idx in block_cover[bi]:
                    tsub_to_blocks[ts_idx].discard(bi)
                block_cover.pop(bi)
                blocks.pop(bi)
                improved = True
                break
        
        if improved:
            continue
        
        # Try to replace a block with a better one (covers more uncovered, or same coverage with different points)
        # Strategy: for each block, try replacing one point with another to see if we can merge coverage
        # Simpler: try pairwise swap - replace block i with a new block that covers all t-subsets 
        # currently only covered by block i, plus maybe more.
        
        # For now, try a simpler heuristic: for each pair of blocks, try to merge them
        # into one block that covers the union of their unique t-subsets.
        # This is complex; instead, try random restarts of greedy on subsets.
        
        # Alternative: try to find a block that can be replaced by a different block
        # that covers the same critical t-subsets.
        # Given time constraints, let's try a different approach:
        # For each block, try all possible k-subsets that contain its critical t-subsets.
        
        # Actually, let's try a simpler local search: for each block, try replacing
        # one of its points with a different point and see if coverage is maintained.
        for bi in range(len(blocks)):
            if (time.time() - start) > time_limit:
                break
            block = blocks[bi]
            # Try replacing each point with each other point
            for pi in range(k):
                old_point = block[pi]
                for new_point in range(v):
                    if new_point == old_point or new_point in block:
                        continue
                    # Try the swap
                    new_block = block[:]
                    new_block[pi] = new_point
                    new_block.sort()
                    
                    # Check if new_block covers all t-subsets that old block uniquely covers
                    # and doesn't lose coverage
                    new_covered = set()
                    for ts in _tsubsets_of_block(new_block, t):
                        idx = tsub_index.get(ts, -1)
                        if idx >= 0:
                            new_covered.add(idx)
                    
                    # Check if all t-subsets covered by old block are still covered
                    ok = True
                    for ts_idx in block_cover[bi]:
                        if ts_idx not in new_covered:
                            # Is it covered by another block?
                            covers = tsub_to_blocks[ts_idx]
                            covers.discard(bi)
                            if not covers:
                                ok = False
                                break
                    if ok:
                        # Perform the swap
                        # Update tsub_to_blocks
                        for ts_idx in block_cover[bi]:
                            tsub_to_blocks[ts_idx].discard(bi)
                        for ts_idx in new_covered:
                            tsub_to_blocks[ts_idx].add(bi)
                        block_cover[bi] = new_covered
                        blocks[bi] = new_block
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return blocks


def construct(v, k, t):
    start_time = time.time()
    rng = random.Random(42)
    
    # Phase 1: Greedy construction
    greedy_time = min(30, _time_left() * 0.3)
    blocks = _greedy_construct(v, k, t, rng, greedy_time)
    
    if blocks is None:
        # Fallback: simple greedy without numpy
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    for x in range(v):
                        if x not in block:
                            best_x = x
                            break
                block.append(best_x)
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
    
    # Verify validity
    if not _is_valid_cover(blocks, v, t):
        # Force validity with a complete fallback
        uncovered = set(itertools.combinations(range(v), t))
        for ts in sorted(uncovered):
            blocks.append(list(ts) + [p for p in range(v) if p not in ts][:k - t])
        # Deduplicate
        seen = set()
        unique_blocks = []
        for b in blocks:
            bt = tuple(sorted(b))
            if bt not in seen:
                seen.add(bt)
                unique_blocks.append(list(bt))
        blocks = unique_blocks
    
    # Phase 2: Local search
    remaining = _time_left()
    if remaining > 5:
        ls_time = min(remaining - 5, 100)
        blocks = _local_search(blocks, v, k, t, rng, ls_time)
    
    # Final validation
    if not _is_valid_cover(blocks, v, t):
        # Emergency fix
        uncovered = set(itertools.combinations(range(v), t))
        for ts in sorted(uncovered):
            # Add a block containing this t-subset
            block = list(ts)
            for p in range(v):
                if p not in block and len(block) < k:
                    block.append(p)
            block.sort()
            if tuple(block) not in set(tuple(sorted(b)) for b in blocks):
                blocks.append(block)
    
    # Deduplicate
    seen = set()
    unique_blocks = []
    for b in blocks:
        bt = tuple(sorted(b))
        if bt not in seen:
            seen.add(bt)
            unique_blocks.append(list(bt))
    
    return unique_blocks
