"""Structural change: vectorized greedy with NumPy bitmasks for fast gain computation,
randomized restarts, and aggressive block-replacement local search.
"""
import time
import itertools
import random
import numpy as np

_DEADLINE = 145.0


def _time_left():
    return _DEADLINE - time.time()


def _all_tsubsets(v, t):
    return list(itertools.combinations(range(v), t))


def _tsubsets_of_block(block, t):
    return list(itertools.combinations(block, t))


def _build_tsub_index(v, t):
    all_tsubs = _all_tsubsets(v, t)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    return all_tsubs, tsub_index, len(all_tsubs)


def _greedy_vectorized(v, k, t, rng, time_limit):
    """Vectorized greedy set cover using NumPy for fast gain computation."""
    all_tsubs, tsub_index, n_tsubs = _build_tsub_index(v, t)
    
    # Precompute for each point p, the bitmask of t-subsets containing p
    # and for each pair (block_prefix, candidate), compute gain quickly
    # 
    # Strategy: represent uncovered t-subsets as a boolean array.
    # For a partial block B, a t-subset ts is coverable by B∪{x} iff
    # ts ⊆ B∪{x} and x ∈ ts.
    # i.e., ts \ {x} ⊆ B.
    
    uncovered = np.ones(n_tsubs, dtype=bool)
    blocks = []
    
    # Precompute: for each point x, list of t-subset indices that contain x
    point_tsubs = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubs):
        for p in ts:
            point_tsubs[p].append(i)
    
    # Precompute: for each t-subset, its elements as a tuple
    # Already have all_tsubs as list of tuples
    
    start = time.time()
    while uncovered.any() and (time.time() - start) < time_limit:
        # Pick anchor: first uncovered t-subset
        anchor_idx = int(np.argmax(uncovered))
        anchor_tsub = all_tsubs[anchor_idx]
        
        block = list(anchor_tsub)
        block_set = set(block)
        
        # Greedily add points
        for _ in range(k - t):
            best_point = -1
            best_gain = -1
            
            # For each candidate point not in block, compute gain
            # Gain = number of uncovered t-subsets ts such that ts ⊆ block∪{p} and p ∈ ts
            #       = number of uncovered t-subsets ts with p∈ts and ts\{p} ⊆ block
            
            # Vectorized approach:
            # For candidate p, the t-subsets containing p are point_tsubs[p]
            # Among those, we need ts\{p} ⊆ block, i.e., all other t-1 elements of ts are in block
            # 
            # Precompute for each t-subset index i: the set of points in it (as a frozenset or bitmask)
            # Use a bitmask representation: each t-subset has a v-bit mask
            
            # Let's precompute t-subset masks once
            # Actually, let's do this inline but efficiently
            
            candidates = [p for p in range(v) if p not in block_set]
            if not candidates:
                break
            
            # For each candidate, compute gain
            # ts\{p} ⊆ block means: for all q in ts, q≠p => q in block
            # Equivalently: ts ⊆ block ∪ {p} and p ∈ ts
            
            gains = np.zeros(len(candidates), dtype=np.int32)
            for ci, p in enumerate(candidates):
                # Get all t-subset indices containing p
                tsub_indices = point_tsubs[p]
                if not tsub_indices:
                    continue
                # For each such t-subset, check if ts\{p} ⊆ block
                # ts\{p} has t-1 elements, all must be in block_set
                # Vectorize: for each t-subset index, check membership
                tsub_arr = np.array(tsub_indices)
                # For each t-subset, get its elements
                # This is still a loop, but let's use the precomputed structure
                # Alternative: precompute for each point p and each (t-1)-subset S of other points,
                # the t-subset index of S∪{p}
                # 
                # Simpler: just loop through point_tsubs[p] and check
                gain = 0
                for ts_idx in tsub_indices:
                    if not uncovered[ts_idx]:
                        continue
                    ts = all_tsubs[ts_idx]
                    # Check if all elements except p are in block
                    ok = True
                    for el in ts:
                        if el != p and el not in block_set:
                            ok = False
                            break
                    if ok:
                        gain += 1
                gains[ci] = gain
            
            if gains.max() > 0:
                best_ci = int(np.argmax(gains))
                best_point = candidates[best_ci]
                best_gain = int(gains[best_ci])
            
            if best_point == -1 or best_gain <= 0:
                # Pick any remaining point
                best_point = candidates[0]
            
            block.append(best_point)
            block_set.add(best_point)
        
        block.sort()
        blocks.append(tuple(block))
        
        # Update uncovered
        for ts in _tsubsets_of_block(block, t):
            idx = tsub_index.get(ts, -1)
            if idx >= 0:
                uncovered[idx] = False
    
    if not uncovered.any():
        return [list(b) for b in blocks]
    return None


def _greedy_simple(v, k, t, rng, time_limit):
    """Simple greedy as fallback."""
    all_tsubs, tsub_index, n_tsubs = _build_tsub_index(v, t)
    uncovered = set(range(n_tsubs))
    blocks = []
    
    start = time.time()
    while uncovered and (time.time() - start) < time_limit:
        anchor = min(uncovered)
        block = list(all_tsubs[anchor])
        block_set = set(block)
        
        while len(block) < k:
            best_point = -1
            best_gain = -1
            for p in range(v):
                if p in block_set:
                    continue
                gain = 0
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
                for p in range(v):
                    if p not in block_set:
                        best_point = p
                        break
                if best_point == -1:
                    break
            block.append(best_point)
            block_set.add(best_point)
        
        block.sort()
        blocks.append(tuple(block))
        for ts in _tsubsets_of_block(block, t):
            idx = tsub_index.get(ts, -1)
            if idx >= 0:
                uncovered.discard(idx)
    
    if not uncovered:
        return [list(b) for b in blocks]
    return None


def _remove_redundant(blocks, v, t):
    """Remove redundant blocks."""
    if not blocks:
        return blocks
    
    all_tsubs, tsub_index, n_tsubs = _build_tsub_index(v, t)
    
    # Build tsub_to_blocks mapping
    tsub_to_blocks = [set() for _ in range(n_tsubs)]
    block_covers = []
    for bi, block in enumerate(blocks):
        covered = set()
        for ts in _tsubsets_of_block(block, t):
            idx = tsub_index.get(ts, -1)
            if idx >= 0:
                covered.add(idx)
                tsub_to_blocks[idx].add(bi)
        block_covers.append(covered)
    
    changed = True
    while changed:
        changed = False
        for bi in range(len(blocks)):
            redundant = True
            for ts_idx in block_covers[bi]:
                if len(tsub_to_blocks[ts_idx]) == 1:
                    redundant = False
                    break
            if redundant:
                for ts_idx in block_covers[bi]:
                    tsub_to_blocks[ts_idx].discard(bi)
                block_covers.pop(bi)
                blocks.pop(bi)
                changed = True
                break
    
    return blocks


def _local_search(blocks, v, k, t, rng, time_limit):
    """Aggressive local search with block replacement."""
    blocks = [list(b) for b in blocks]
    
    all_tsubs, tsub_index, n_tsubs = _build_tsub_index(v, t)
    
    # Build coverage structures
    tsub_to_blocks = [set() for _ in range(n_tsubs)]
    block_covers = []
    for bi, block in enumerate(blocks):
        covered = set()
        for ts in _tsubsets_of_block(block, t):
            idx = tsub_index.get(ts, -1)
            if idx >= 0:
                covered.add(idx)
                tsub_to_blocks[idx].add(bi)
        block_covers.append(covered)
    
    start = time.time()
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        
        # Phase 1: Remove redundant blocks
        for bi in range(len(blocks)):
            if (time.time() - start) > time_limit:
                break
            redundant = True
            for ts_idx in block_covers[bi]:
                if len(tsub_to_blocks[ts_idx]) == 1:
                    redundant = False
                    break
            if redundant:
                for ts_idx in block_covers[bi]:
                    tsub_to_blocks[ts_idx].discard(bi)
                block_covers.pop(bi)
                blocks.pop(bi)
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: Try point swaps
        for bi in range(len(blocks)):
            if (time.time() - start) > time_limit:
                break
            block = blocks[bi]
            for pi in range(k):
                old_point = block[pi]
                for new_point in range(v):
                    if new_point == old_point or new_point in block:
                        continue
                    new_block = block[:]
                    new_block[pi] = new_point
                    new_block.sort()
                    
                    new_covered = set()
                    for ts in _tsubsets_of_block(new_block, t):
                        idx = tsub_index.get(ts, -1)
                        if idx >= 0:
                            new_covered.add(idx)
                    
                    ok = True
                    for ts_idx in block_covers[bi]:
                        if ts_idx not in new_covered:
                            covers = tsub_to_blocks[ts_idx]
                            covers.discard(bi)
                            if not covers:
                                ok = False
                                break
                    if ok:
                        for ts_idx in block_covers[bi]:
                            tsub_to_blocks[ts_idx].discard(bi)
                        for ts_idx in new_covered:
                            tsub_to_blocks[ts_idx].add(bi)
                        block_covers[bi] = new_covered
                        blocks[bi] = new_block
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        # Phase 3: Try replacing a block entirely
        # For each block, find its critical t-subsets (only covered by this block)
        # Then try to find a new block that covers all critical t-subsets
        for bi in range(len(blocks)):
            if (time.time() - start) > time_limit:
                break
            critical = [ts_idx for ts_idx in block_covers[bi] 
                       if len(tsub_to_blocks[ts_idx]) == 1]
            if not critical:
                continue
            
            # The new block must contain all critical t-subsets
            # Find the intersection of points that must be in the new block
            # Each critical t-subset must be contained in the new block
            # So the new block must contain the union of all critical t-subsets' points? No,
            # each critical t-subset individually must be a subset of the new block.
            # So the new block must contain each critical t-subset as a subset.
            # This means the new block must contain the union of all points in critical t-subsets
            # if k >= t, but the union might be larger than k.
            
            # Collect all points that appear in any critical t-subset
            required_points = set()
            for ts_idx in critical:
                for p in all_tsubs[ts_idx]:
                    required_points.add(p)
            
            if len(required_points) > k:
                continue
            
            # Try to complete required_points to a k-set
            if len(required_points) == k:
                candidate = sorted(required_points)
            else:
                # Pick random additional points
                remaining = [p for p in range(v) if p not in required_points]
                if not remaining:
                    continue
                extra_needed = k - len(required_points)
                extras = rng.sample(remaining, min(extra_needed, len(remaining)))
                candidate = sorted(required_points | set(extras))
                if len(candidate) < k:
                    continue
            
            # Check if this candidate covers all critical t-subsets
            candidate_set = set(candidate)
            valid = True
            for ts_idx in critical:
                ts = all_tsubs[ts_idx]
                if not all(p in candidate_set for p in ts):
                    valid = False
                    break
            if not valid:
                continue
            
            # Compute new coverage
            new_covered = set()
            for ts in _tsubsets_of_block(candidate, t):
                idx = tsub_index.get(ts, -1)
                if idx >= 0:
                    new_covered.add(idx)
            
            # Check no coverage is lost
            ok = True
            for ts_idx in block_covers[bi]:
                if ts_idx not in new_covered:
                    covers = tsub_to_blocks[ts_idx]
                    covers.discard(bi)
                    if not covers:
                        ok = False
                        break
            if not ok:
                continue
            
            # Perform replacement
            for ts_idx in block_covers[bi]:
                tsub_to_blocks[ts_idx].discard(bi)
            for ts_idx in new_covered:
                tsub_to_blocks[ts_idx].add(bi)
            block_covers[bi] = new_covered
            blocks[bi] = candidate
            improved = True
            break
    
    return blocks


def _is_valid_cover(blocks, v, t):
    all_tsubs, tsub_index, n_tsubs = _build_tsub_index(v, t)
    covered = set()
    for block in blocks:
        for ts in _tsubsets_of_block(block, t):
            idx = tsub_index.get(ts, -1)
            if idx >= 0:
                covered.add(idx)
    return len(covered) == n_tsubs


def construct(v, k, t):
    start_time = time.time()
    rng = random.Random(42)
    
    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Try multiple greedy strategies
    strategies = []
    
    # Strategy 1: Vectorized greedy
    t1 = min(20, _time_left() * 0.2)
    blocks1 = _greedy_vectorized(v, k, t, rng, t1)
    if blocks1:
        strategies.append(blocks1)
    
    # Strategy 2: Simple greedy
    if _time_left() > 5:
        t2 = min(15, _time_left() * 0.15)
        blocks2 = _greedy_simple(v, k, t, rng, t2)
        if blocks2:
            strategies.append(blocks2)
    
    if not strategies:
        # Emergency fallback
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            ts = min(uncovered)
            block = list(ts)
            for p in range(v):
                if p not in block and len(block) < k:
                    block.append(p)
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        strategies.append(blocks)
    
    # Pick the best initial solution
    for blocks in strategies:
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    if best_blocks is None:
        # Should not happen
        return []
    
    # Phase 2: Local search
    remaining = _time_left()
    if remaining > 10:
        ls_time = min(remaining - 5, 100)
        # Try local search on the best solution
        ls_blocks = _local_search(best_blocks, v, k, t, rng, ls_time)
        if ls_blocks and len(ls_blocks) < best_count:
            best_blocks = ls_blocks
            best_count = len(ls_blocks)
    
    # Final validation
    if not _is_valid_cover(best_blocks, v, t):
        # Emergency fix
        all_tsubs, tsub_index, n_tsubs = _build_tsub_index(v, t)
        covered = set()
        for block in best_blocks:
            for ts in _tsubsets_of_block(block, t):
                idx = tsub_index.get(ts, -1)
                if idx >= 0:
                    covered.add(idx)
        for i in range(n_tsubs):
            if i not in covered:
                ts = all_tsubs[i]
                block = list(ts)
                for p in range(v):
                    if p not in block and len(block) < k:
                        block.append(p)
                block.sort()
                if tuple(block) not in set(tuple(sorted(b)) for b in best_blocks):
                    best_blocks.append(block)
    
    # Deduplicate
    seen = set()
    unique_blocks = []
    for b in best_blocks:
        bt = tuple(sorted(b))
        if bt not in seen:
            seen.add(bt)
            unique_blocks.append(list(bt))
    
    return unique_blocks
