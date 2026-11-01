"""Covering design C(24,7,3): fast randomized greedy with local search reduction."""
import itertools
import time
import random
from collections import defaultdict


def _fast_greedy(v, k, t, rng, time_limit):
    """
    Fast greedy covering with vectorized gain computation.
    Uses numpy for speed where possible.
    """
    import numpy as np
    
    deadline = time.time() + time_limit
    all_tsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsets)
    blocks = []
    
    # Precompute: for each (t-1)-subset, which t-sets contain it
    # This allows fast gain computation
    tm1_to_tsets = defaultdict(set)
    for ts in all_tsets:
        for i in range(t):
            tm1 = ts[:i] + ts[i+1:]
            tm1_to_tsets[tm1].add(ts)
    
    while uncovered and time.time() < deadline:
        # Pick a random uncovered t-set to start the block
        start_tset = rng.choice(list(uncovered))
        block = set(start_tset)
        
        while len(block) < k:
            # Compute gain for each candidate point
            best_point = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    new_tset = tuple(sorted(rest + (x,)))
                    if new_tset in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point == -1:
                break
            block.add(best_point)
        
        block_list = sorted(block)
        blocks.append(block_list)
        # Remove covered t-sets
        for ts in itertools.combinations(block_list, t):
            uncovered.discard(ts)
    
    return blocks


def _remove_redundant_blocks(blocks, v, k, t):
    """Remove blocks that don't cover any unique t-set."""
    import numpy as np
    
    all_tsets = list(itertools.combinations(range(v), t))
    tset_index = {ts: i for i, ts in enumerate(all_tsets)}
    n_tsets = len(all_tsets)
    n_blocks = len(blocks)
    
    # Build coverage matrix: block b covers t-set s
    cover = np.zeros((n_blocks, n_tsets), dtype=np.int8)
    for b, block in enumerate(blocks):
        for ts in itertools.combinations(block, t):
            cover[b, tset_index[ts]] = 1
    
    # For each block, check if removing it leaves all t-sets covered
    result = []
    for b in range(n_blocks):
        # Temporarily remove block b
        covered_without = np.sum(cover, axis=0) - cover[b]
        if np.all(covered_without >= 1):
            # Block b is redundant, skip it
            continue
        result.append(blocks[b])
    
    return result


def _local_search(blocks, v, k, t, rng, time_limit):
    """Local search: try to replace blocks to reduce count."""
    deadline = time.time() + time_limit
    
    all_tsets = list(itertools.combinations(range(v), t))
    tset_index = {ts: i for i, ts in enumerate(all_tsets)}
    n_tsets = len(all_tsets)
    
    def is_valid(blks):
        covered = set()
        for block in blks:
            for ts in itertools.combinations(block, t):
                covered.add(ts)
        return len(covered) == n_tsets
    
    current = list(blocks)
    
    while time.time() < deadline:
        improved = False
        n = len(current)
        
        # Try removing each block and checking validity
        for i in range(n):
            if time.time() >= deadline:
                break
            trial = current[:i] + current[i+1:]
            if is_valid(trial):
                current = trial
                improved = True
                break
        
        if not improved:
            # Try replacing a block with a better one
            for i in range(n):
                if time.time() >= deadline:
                    break
                
                # Find t-sets only covered by block i
                covered_by_others = set()
                for j, block in enumerate(current):
                    if j == i:
                        continue
                    for ts in itertools.combinations(block, t):
                        covered_by_others.add(ts)
                
                unique_to_i = set()
                for ts in itertools.combinations(current[i], t):
                    if ts not in covered_by_others:
                        unique_to_i.add(ts)
                
                if not unique_to_i:
                    continue
                
                # Try to find a new block that covers all unique_to_i t-sets
                # and is different from existing blocks
                existing_set = set(tuple(b) for b in current)
                
                # Greedily build a block that covers unique_to_i
                must_cover = set(unique_to_i)
                new_block = set()
                
                # Start with a point from a unique t-set
                if must_cover:
                    start_ts = next(iter(must_cover))
                    new_block.add(start_ts[0])
                
                while len(new_block) < k and must_cover:
                    best_point = -1
                    best_gain = -1
                    for x in range(v):
                        if x in new_block:
                            continue
                        gain = 0
                        for ts in must_cover:
                            if x in ts and all(p in new_block or p == x for p in ts):
                                # Check if adding x helps cover ts
                                pass
                        # Simpler: count how many must_cover t-sets become fully covered
                        for ts in must_cover:
                            if x in ts:
                                remaining = set(ts) - new_block - {x}
                                if not remaining:
                                    gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_point = x
                    if best_point == -1:
                        break
                    new_block.add(best_point)
                
                new_block_list = sorted(new_block)
                if tuple(new_block_list) not in existing_set:
                    # Check if replacing block i with new_block_list is valid
                    trial = current[:i] + [new_block_list] + current[i+1:]
                    if is_valid(trial):
                        current = trial
                        improved = True
                        break
        
        if not improved:
            break
    
    return current


def construct(v, k, t):
    """
    Construct a covering design with as few blocks as possible.
    """
    start_time = time.time()
    total_time_limit = 110  # Leave 10s margin for the 120s limit
    
    rng = random.Random(42)
    
    # Phase 1: Fast greedy with multiple seeds
    best_blocks = None
    best_count = float('inf')
    
    # Try a few different random seeds for the greedy
    for seed in range(5):
        if time.time() - start_time > total_time_limit * 0.5:
            break
        rng = random.Random(seed)
        blocks = _fast_greedy(v, k, t, rng, time_limit=total_time_limit * 0.1)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    if best_blocks is None:
        # Fallback: simple greedy
        blocks = _fast_greedy(v, k, t, random.Random(0), time_limit=total_time_limit * 0.5)
        best_blocks = blocks
        best_count = len(blocks)
    
    # Phase 2: Remove redundant blocks
    if time.time() - start_time < total_time_limit * 0.7:
        best_blocks = _remove_redundant_blocks(best_blocks, v, k, t)
        best_count = len(best_blocks)
    
    # Phase 3: Local search
    remaining_time = total_time_limit - (time.time() - start_time)
    if remaining_time > 5:
        best_blocks = _local_search(best_blocks, v, k, t, rng, time_limit=remaining_time * 0.8)
    
    return best_blocks
