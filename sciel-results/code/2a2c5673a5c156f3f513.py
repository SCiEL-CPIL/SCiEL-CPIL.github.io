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
        # Pick a random uncovered t-set to start
        start = rng.choice(list(uncovered))
        block = list(start)
        block_set = set(block)
        
        while len(block) < k:
            if time.time() > deadline:
                break
            
            best_x = -1
            best_gain = -1
            
            # Compute gain for each candidate
            for x in range(v):
                if x in block_set:
                    continue
                
                gain = 0
                for S in itertools.combinations(block, t-1):
                    tset = tuple(sorted(S + (x,)))
                    if tset in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x == -1:
                break
            
            block.append(best_x)
            block_set.add(best_x)
        
        if len(block) < k:
            break
        
        block.sort()
        blocks.append(tuple(block))
        
        # Remove covered t-sets
        for ts in itertools.combinations(block, t):
            uncovered.discard(ts)
    
    return [list(b) for b in blocks]

def _local_search_reduce(blocks, v, k, t, time_limit):
    """
    Reduce blocks by removing redundant ones.
    A block is redundant if no t-set is covered only by it.
    """
    deadline = time.time() + time_limit
    blocks = [list(b) for b in blocks]
    
    all_tsets = list(itertools.combinations(range(v), t))
    tset_to_blocks = defaultdict(list)
    
    def build_indices(blocks):
        tset_to_blocks.clear()
        for i, b in enumerate(blocks):
            for ts in itertools.combinations(b, t):
                tset_to_blocks[ts].append(i)
        return tset_to_blocks
    
    build_indices(blocks)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(blocks)):
            if time.time() > deadline:
                return [list(b) for b in blocks]
            
            b = blocks[i]
            # Check if any t-set is covered only by block i
            only_covered = False
            for ts in itertools.combinations(b, t):
                if len(tset_to_blocks[ts]) == 1:
                    only_covered = True
                    break
            
            if not only_covered:
                # Remove block i
                blocks.pop(i)
                build_indices(blocks)
                improved = True
                break
    
    return blocks

def _swap_improve(blocks, v, k, t, rng, time_limit):
    """
    Try to improve by swapping blocks.
    """
    deadline = time.time() + time_limit
    blocks = [list(b) for b in blocks]
    
    all_tsets = list(itertools.combinations(range(v), t))
    
    def build_indices(blocks):
        tset_to_blocks = defaultdict(list)
        for i, b in enumerate(blocks):
            for ts in itertools.combinations(b, t):
                tset_to_blocks[ts].append(i)
        return tset_to_blocks
    
    tset_to_blocks = build_indices(blocks)
    
    for _ in range(100):
        if time.time() > deadline:
            break
        
        n = len(blocks)
        if n == 0:
            break
        
        idx = rng.randint(0, n - 1)
        old_block = blocks[idx]
        old_set = set(old_block)
        
        # Find critical t-sets (covered only by this block)
        critical = []
        for ts in itertools.combinations(old_block, t):
            if len(tset_to_blocks[ts]) == 1:
                critical.append(ts)
        
        if not critical:
            # Redundant, remove
            blocks.pop(idx)
            tset_to_blocks = build_indices(blocks)
            continue
        
        # Try random replacements
        for _ in range(20):
            if time.time() > deadline:
                break
            
            cand = sorted(rng.sample(range(v), k))
            cand_set = set(cand)
            
            # Check if cand covers all critical t-sets
            covers_all = True
            for ts in critical:
                if not set(ts).issubset(cand_set):
                    covers_all = False
                    break
            
            if covers_all:
                # Also check that removing old_block doesn't uncover anything else
                # (i.e., all t-sets in old_block are covered by cand or other blocks)
                valid = True
                for ts in itertools.combinations(old_block, t):
                    if not set(ts).issubset(cand_set):
                        # Check if covered by another block
                        covered = False
                        for j, b in enumerate(blocks):
                            if j != idx and set(ts).issubset(set(b)):
                                covered = True
                                break
                        if not covered:
                            valid = False
                            break
                
                if valid:
                    blocks[idx] = cand
                    tset_to_blocks = build_indices(blocks)
                    break
    
    return blocks

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using randomized greedy with local search.
    """
    start_time = time.time()
    total_time_limit = 110  # Leave 10s margin before 120s limit
    
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple greedy instances with different random seeds
    rng = random.Random(42)
    
    # First, run a few fast greedy instances
    num_greedy = 5
    greedy_time_each = (total_time_limit * 0.4) / num_greedy
    
    for i in range(num_greedy):
        if time.time() - start_time > total_time_limit * 0.4:
            break
        
        blocks = _fast_greedy(v, k, t, rng, greedy_time_each)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    if best_blocks is None:
        # Fallback: simple greedy
        blocks = _fast_greedy(v, k, t, rng, total_time_limit)
        return blocks
    
    # Local search reduction
    remaining_time = total_time_limit - (time.time() - start_time)
    
    # First pass: remove redundant blocks
    reduced = _local_search_reduce(best_blocks, v, k, t, remaining_time * 0.3)
    
    if len(reduced) < best_len:
        best_len = len(reduced)
        best_blocks = reduced
    
    # Second pass: swap improvement
    remaining_time = total_time_limit - (time.time() - start_time)
    if remaining_time > 5:
        swapped = _swap_improve(best_blocks, v, k, t, rng, remaining_time * 0.5)
        
        if len(swapped) < best_len:
            best_len = len(swapped)
            best_blocks = swapped
    
    # Final reduction pass
    remaining_time = total_time_limit - (time.time() - start_time)
    if remaining_time > 2:
        final = _local_search_reduce(best_blocks, v, k, t, remaining_time * 0.5)
        
        if len(final) < best_len:
            best_len = len(final)
            best_blocks = final
    
    # Ensure we return within time limit
    if time.time() - start_time > total_time_limit:
        return best_blocks
    
    return best_blocks
