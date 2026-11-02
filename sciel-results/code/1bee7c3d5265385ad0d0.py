"""
Covering design C(24,7,3) via greedy set-cover with redundancy removal and local search.
"""
import itertools
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def _build_block_index(blocks, v, t):
    """For each t-subset, store list of block indices that contain it."""
    idx = defaultdict(list)
    for bi, block in enumerate(blocks):
        for comb in itertools.combinations(block, t):
            idx[comb].append(bi)
    return idx

def _remove_redundant_blocks(blocks, v, t, time_limit=None):
    """Remove blocks that are not needed to cover all t-subsets."""
    if not blocks:
        return blocks
    n = len(blocks)
    # Build index: t-subset -> set of block indices
    idx = defaultdict(set)
    for bi, block in enumerate(blocks):
        for comb in itertools.combinations(block, t):
            idx[comb].add(bi)
    
    # Count how many t-subsets each block uniquely covers
    unique_count = [0] * n
    for comb, bset in idx.items():
        if len(bset) == 1:
            unique_count[list(bset)[0]] += 1
    
    # Iteratively remove non-unique blocks
    removed = [False] * n
    changed = True
    while changed:
        changed = False
        for bi in range(n):
            if removed[bi]:
                continue
            if unique_count[bi] == 0:
                # This block is not uniquely covering anything; try removing it
                # Verify all its t-subsets are still covered by other blocks
                safe = True
                for comb in itertools.combinations(blocks[bi], t):
                    others = idx[comb] - {bi}
                    if not others:
                        safe = False
                        break
                if safe:
                    removed[bi] = True
                    changed = True
                    # Update unique counts
                    for comb in itertools.combinations(blocks[bi], t):
                        idx[comb].discard(bi)
                        if len(idx[comb]) == 1:
                            unique_count[list(idx[comb])[0]] += 1
                        elif len(idx[comb]) == 0:
                            # Shouldn't happen if safe, but guard
                            pass
                    # Decrement unique counts for blocks that lost their unique coverage
                    # Actually we need to recompute; let's just mark and continue
                    # For efficiency, just rebuild if needed
                    pass
        if time_limit and time.time() > time_limit:
            break
    
    # Rebuild properly
    kept = [blocks[i] for i in range(n) if not removed[i]]
    return kept

def _greedy_construct(v, k, t, rng=None):
    """Greedy set-cover: repeatedly pick the block covering most uncovered t-subsets."""
    if rng is None:
        rng = random.Random(42)
    
    all_points = list(range(v))
    all_tsubs = set(itertools.combinations(all_points, t))
    uncovered = set(all_tsubs)
    blocks = []
    
    while uncovered:
        # Find block that covers most uncovered t-subsets
        best_block = None
        best_count = -1
        best_gain = None
        
        # Sample candidate blocks to keep it fast
        # Generate random blocks and pick the best
        candidates = []
        for _ in range(200):
            block = tuple(sorted(rng.sample(all_points, k)))
            gain = sum(1 for comb in itertools.combinations(block, t) if comb in uncovered)
            candidates.append((gain, block))
        
        candidates.sort(reverse=True)
        best_gain, best_block = candidates[0]
        
        if best_gain <= 0:
            # Fallback: pick any block
            block = tuple(sorted(rng.sample(all_points, k)))
            blocks.append(list(block))
            for comb in itertools.combinations(block, t):
                uncovered.discard(comb)
            continue
        
        blocks.append(list(best_block))
        for comb in itertools.combinations(best_block, t):
            uncovered.discard(comb)
    
    return blocks

def _local_search(blocks, v, k, t, max_iter=200, rng=None):
    """Try to reduce block count by replacing pairs or removing/re-adding."""
    if rng is None:
        rng = random.Random(123)
    
    blocks = [list(b) for b in blocks]
    n = len(blocks)
    
    # Build coverage index
    def build_index(blks):
        idx = defaultdict(set)
        for bi, block in enumerate(blks):
            for comb in itertools.combinations(block, t):
                idx[comb].add(bi)
        return idx
    
    idx = build_index(blocks)
    
    for _ in range(max_iter):
        # Try removing a block and see if we can repair with one new block
        bi = rng.randrange(len(blocks))
        candidate = blocks[bi]
        # Check if removing it leaves any uncovered t-subset
        orphaned = []
        for comb in itertools.combinations(candidate, t):
            if len(idx[comb]) == 1:
                orphaned.append(comb)
        
        if not orphaned:
            # Safe to remove
            blocks.pop(bi)
            # Rebuild index
            idx = build_index(blocks)
            continue
        
        # Try to find a block that covers all orphaned t-subsets
        # The orphaned t-subsets must all be contained in some k-subset
        # Collect all points involved
        points_needed = set()
        for comb in orphaned:
            points_needed.update(comb)
        
        if len(points_needed) > k:
            continue
        
        # Try extending points_needed to a k-subset
        remaining = [p for p in range(v) if p not in points_needed]
        need_more = k - len(points_needed)
        if need_more > 0:
            extra = rng.sample(remaining, need_more)
        else:
            extra = []
        
        new_block = tuple(sorted(list(points_needed) + extra))
        
        # Check this new block covers all orphaned
        new_tsubs = set(itertools.combinations(new_block, t))
        if all(c in new_tsubs for c in orphaned):
            # Replace
            blocks[bi] = list(new_block)
            idx = build_index(blocks)
    
    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for the 120s limit
    
    rng = random.Random(42)
    
    # Phase 1: Greedy construction
    blocks = _greedy_construct(v, k, t, rng)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return blocks
    
    # Phase 2: Remove redundant blocks
    blocks = _remove_redundant_blocks(blocks, v, t, time_limit=deadline - 20)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 15:
        return blocks
    
    # Phase 3: Local search to reduce count
    blocks = _local_search(blocks, v, k, t, max_iter=50, rng=rng)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return blocks
    
    # Phase 4: Another round of redundancy removal
    blocks = _remove_redundant_blocks(blocks, v, t, time_limit=deadline - 2)
    
    return blocks

# EVOLVE-BLOCK-END
