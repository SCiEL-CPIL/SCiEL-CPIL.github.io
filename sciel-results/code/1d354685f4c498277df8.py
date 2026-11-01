import itertools
import time
import random
import sys
import numpy as np

_global_deadline = None

def _precompute(v, k, t):
    """
    Precomputes:
    1. all_blocks: list of all k-subsets of range(v)
    2. block_to_ts: list of lists, where block_to_ts[i] is the list of t-subsets in all_blocks[i]
    3. ts_to_blocks: dict mapping t-subset tuple to list of indices in all_blocks
    4. ts_to_block_set: dict mapping t-subset tuple to set of indices (for faster membership)
    """
    all_blocks = []
    block_to_ts = []
    ts_to_blocks = {}
    ts_to_block_set = {}
    
    # Generate all k-subsets
    for block in itertools.combinations(range(v), k):
        idx = len(all_blocks)
        all_blocks.append(block)
        
        # Generate all t-subsets of this block
        t_subs = list(itertools.combinations(block, t))
        block_to_ts.append(t_subs)
        
        for ts in t_subs:
            ts_tuple = tuple(sorted(ts))
            if ts_tuple not in ts_to_blocks:
                ts_to_blocks[ts_tuple] = []
                ts_to_block_set[ts_tuple] = set()
            ts_to_blocks[ts_tuple].append(idx)
            ts_to_block_set[ts_tuple].add(idx)
            
    return all_blocks, block_to_ts, ts_to_blocks, ts_to_block_set

def _greedy_cover_rarest(v, k, t, all_blocks, block_to_ts, ts_to_blocks, ts_to_block_set, deadline):
    """
    Greedy set cover using the 'rarest first' heuristic.
    At each step, find the uncovered t-subset with the fewest candidate blocks,
    then pick the block among those candidates that covers the most uncovered t-subsets.
    """
    # Start with all t-subsets uncovered
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    while uncovered and time.time() < deadline:
        # Find the uncovered t-subset with the minimum number of candidate blocks
        # To avoid scanning all uncovered each time (expensive), we can sample or use a priority queue.
        # For now, let's try a batch approach or a simple scan with early termination if possible.
        # Actually, scanning all uncovered to find min count is O(|uncovered| * lookup).
        # |uncovered| starts at 11628. This is manageable if we do it smartly.
        
        min_count = float('inf')
        seed_ts = None
        
        # Optimization: We don't need to check all uncovered if we find a count of 1 early?
        # No, we want the global minimum to make the best choice.
        # But 11628 iterations per step * ~300 steps = 3.5M lookups. Fast enough.
        
        for ts in uncovered:
            count = len(ts_to_blocks.get(ts, []))
            if count < min_count:
                min_count = count
                seed_ts = ts
                if min_count == 1:
                    break # Can't get better than 1
        
        if seed_ts is None:
            break
            
        candidates = ts_to_blocks.get(seed_ts, [])
        if not candidates:
            break
            
        best_idx = -1
        best_gain = -1
        
        for idx in candidates:
            gain = 0
            for ts in block_to_ts[idx]:
                if ts in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                
        if best_idx < 0:
            break
            
        blocks.append(all_blocks[best_idx])
        
        # Remove covered t-subsets from uncovered
        for ts in block_to_ts[best_idx]:
            uncovered.discard(ts)
            
    return blocks

def _greedy_cover_random(v, k, t, all_blocks, block_to_ts, ts_to_blocks, ts_to_block_set, rng, deadline):
    """
    Standard greedy set cover with random seed selection.
    """
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    while uncovered and time.time() < deadline:
        seed_ts = rng.choice(list(uncovered))
        candidates = ts_to_blocks.get(seed_ts, [])
        if not candidates:
            break
            
        best_idx = -1
        best_gain = -1
        
        for idx in candidates:
            gain = 0
            for ts in block_to_ts[idx]:
                if ts in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                
        if best_idx < 0:
            break
            
        blocks.append(all_blocks[best_idx])
        
        for ts in block_to_ts[best_idx]:
            uncovered.discard(ts)
            
    return blocks

def _local_search(blocks, v, k, t, all_blocks, block_to_ts, ts_to_blocks, ts_to_block_set, deadline):
    """
    Local search with:
    1. Redundancy removal.
    2. 1-for-1 block swaps to reduce count or maintain count while improving structure.
    3. Pairwise merge attempts.
    """
    blocks = [list(b) for b in blocks]
    
    def build_coverage(block_list):
        counts = {}
        for b in block_list:
            b_set = set(b)
            for ts in itertools.combinations(b, t):
                ts_t = tuple(sorted(ts))
                counts[ts_t] = counts.get(ts_t, 0) + 1
        return counts

    coverage = build_coverage(blocks)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Remove redundant blocks
        n = len(blocks)
        for i in range(n):
            if time.time() >= deadline:
                break
            b = blocks[i]
            redundant = True
            for ts in itertools.combinations(b, t):
                ts_t = tuple(sorted(ts))
                if coverage.get(ts_t, 0) == 1:
                    redundant = False
                    break
            if redundant:
                removed = blocks.pop(i)
                for ts in itertools.combinations(removed, t):
                    ts_t = tuple(sorted(ts))
                    coverage[ts_t] -= 1
                    if coverage[ts_t] <= 0:
                        del coverage[ts_t]
                improved = True
                break
        
        if improved:
            continue
            
        # Phase 2: 1-for-1 Swaps
        # Try to replace a block with another that covers its unique parts and adds more new coverage.
        n = len(blocks)
        for i in range(n):
            if time.time() >= deadline:
                break
                
            b_i = blocks[i]
            # Find t-subsets uniquely covered by b_i
            unique_to_i = set()
            for ts in itertools.combinations(b_i, t):
                ts_t = tuple(sorted(ts))
                if coverage.get(ts_t, 0) == 1:
                    unique_to_i.add(ts_t)
                    
            if not unique_to_i:
                # This block is redundant, should have been caught in Phase 1
                # But just in case, remove it? No, Phase 1 handles it.
                continue
                
            # Find candidate blocks that cover all unique_to_i
            # We can intersect the candidate sets for each ts in unique_to_i
            candidates = None
            for ts in unique_to_i:
                cand_set = ts_to_block_set.get(ts, set())
                if candidates is None:
                    candidates = set(cand_set)
                else:
                    candidates &= cand_set
                    
            # Exclude current block i
            candidates.discard(i) # Wait, i is an index in blocks, not all_blocks.
            # We need to map blocks to all_blocks indices?
            # Actually, blocks are specific instances. A block might appear multiple times? No, distinct blocks.
            # But the index in `blocks` is not the index in `all_blocks`.
            # We need to check if a candidate block (from all_blocks) is already in `blocks`.
            # Or better, just iterate through all_blocks candidates and check if they are in `blocks`.
            
            # Optimization: Instead of iterating all candidates in `candidates` (which are all_blocks indices),
            # we can just check if they are valid replacements.
            
            best_swap = None
            best_gain = -1
            
            # We only care about candidates that are not already in blocks
            existing_block_tuples = [tuple(b) for b in blocks]
            
            for cand_idx in candidates:
                cand_block = all_blocks[cand_idx]
                if tuple(cand_block) in existing_block_tuples:
                    continue
                    
                # Calculate gain: how many NEW t-subsets does this block cover?
                gain = 0
                for ts in block_to_ts[cand_idx]:
                    if ts not in coverage or coverage.get(ts, 0) == 0:
                        gain += 1
                        
                # We also need to ensure we don't lose coverage.
                # Since cand_block covers all unique_to_i, removing b_i is safe for those.
                # For non-unique t-subsets in b_i, they are covered by other blocks.
                # So the net change in coverage count is +gain (newly covered).
                
                if gain > best_gain:
                    best_gain = gain
                    best_swap = cand_block
                    
            if best_swap is not None and best_gain > 0:
                # Perform swap
                # Update coverage for removed block
                for ts in itertools.combinations(b_i, t):
                    ts_t = tuple(sorted(ts))
                    coverage[ts_t] -= 1
                    if coverage[ts_t] <= 0:
                        del coverage[ts_t]
                        
                # Update coverage for new block
                for ts in itertools.combinations(best_swap, t):
                    ts_t = tuple(sorted(ts))
                    coverage[ts_t] = coverage.get(ts_t, 0) + 1
                    
                blocks[i] = sorted(best_swap)
                improved = True
                break
                
        if improved:
            continue
            
        # Phase 3: Pairwise Merge (Simplified)
        # Try to replace two blocks with one.
        n = len(blocks)
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                if time.time() >= deadline:
                    break
                    
                b_i = blocks[i]
                b_j = blocks[j]
                
                # Find t-subsets that become uncovered if we remove i and j
                uncovered_set = set()
                
                for ts in itertools.combinations(b_i, t):
                    ts_t = tuple(sorted(ts))
                    c = coverage.get(ts_t, 0)
                    if c == 1:
                        uncovered_set.add(ts_t)
                    elif c == 2:
                        # Check if in b_j
                        if set(ts_t).issubset(set(b_j)):
                            uncovered_set.add(ts_t)
                            
                for ts in itertools.combinations(b_j, t):
                    ts_t = tuple(sorted(ts))
                    c = coverage.get(ts_t, 0)
                    if c == 1:
                        uncovered_set.add(ts_t)
                    elif c == 2:
                        # If in b_i, already added. If not, covered by another.
                        pass

                if not uncovered_set:
                    # Both redundant?
                    blocks.pop(j)
                    blocks.pop(i)
                    # Rebuild coverage? Or update.
                    # Update is faster.
                    for ts in itertools.combinations(b_i, t):
                        ts_t = tuple(sorted(ts))
                        coverage[ts_t] -= 1
                        if coverage[ts_t] <= 0: del coverage[ts_t]
                    for ts in itertools.combinations(b_j, t):
                        ts_t = tuple(sorted(ts))
                        coverage[ts_t] -= 1
                        if coverage[ts_t] <= 0: del coverage[ts_t]
                    improved = True
                    break
                    
                # Check if union fits in k
                union_points = set()
                for ts in uncovered_set:
                    union_points.update(ts)
                    
                if len(union_points) <= k:
                    new_block = list(union_points)
                    remaining = [x for x in range(v) if x not in union_points]
                    if len(new_block) < k:
                        new_block.extend(random.sample(remaining, k - len(new_block)))
                    new_block.sort()
                    
                    # Check if new_block exists
                    found_existing = False
                    for idx, b in enumerate(blocks):
                        if idx == i or idx == j: continue
                        if b == new_block:
                            found_existing = True
                            break
                            
                    if found_existing:
                        # Just remove i and j
                        blocks.pop(j)
                        blocks.pop(i)
                        for ts in itertools.combinations(b_i, t):
                            ts_t = tuple(sorted(ts))
                            coverage[ts_t] -= 1
                            if coverage[ts_t] <= 0: del coverage[ts_t]
                        for ts in itertools.combinations(b_j, t):
                            ts_t = tuple(sorted(ts))
                            coverage[ts_t] -= 1
                            if coverage[ts_t] <= 0: del coverage[ts_t]
                    else:
                        # Remove i and j, add new_block
                        blocks.pop(j)
                        blocks.pop(i)
                        blocks.append(new_block)
                        
                        # Update coverage
                        for ts in itertools.combinations(b_i, t):
                            ts_t = tuple(sorted(ts))
                            coverage[ts_t] -= 1
                            if coverage[ts_t] <= 0: del coverage[ts_t]
                        for ts in itertools.combinations(b_j, t):
                            ts_t = tuple(sorted(ts))
                            coverage[ts_t] -= 1
                            if coverage[ts_t] <= 0: del coverage[ts_t]
                        for ts in itertools.combinations(new_block, t):
                            ts_t = tuple(sorted(ts))
                            coverage[ts_t] = coverage.get(ts_t, 0) + 1
                            
                    improved = True
                    break
                    
            if improved:
                break
                
    return blocks

def construct(v, k, t):
    global _global_deadline
    _global_deadline = time.time() + 115
    rng = random.Random(42)
    
    # Precompute
    all_blocks, block_to_ts, ts_to_blocks, ts_to_block_set = _precompute(v, k, t)
    
    best_blocks = None
    best_count = float('inf')
    
    # Alternate between rarest-first and random greedy
    # Rarest-first is often stronger for covering designs
    num_restarts = 6
    
    for restart in range(num_restarts):
        if time.time() > _global_deadline - 25:
            break
            
        if restart % 2 == 0:
            blocks = _greedy_cover_rarest(v, k, t, all_blocks, block_to_ts, ts_to_blocks, ts_to_block_set, _global_deadline - 25)
        else:
            blocks = _greedy_cover_random(v, k, t, all_blocks, block_to_ts, ts_to_blocks, ts_to_block_set, rng, _global_deadline - 25)
        
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
            
        # Local search
        if time.time() < _global_deadline - 10:
            improved_blocks = _local_search(blocks, v, k, t, all_blocks, block_to_ts, ts_to_blocks, ts_to_block_set, _global_deadline - 10)
            if len(improved_blocks) < best_count:
                best_blocks = improved_blocks
                best_count = len(improved_blocks)
                
    return best_blocks
