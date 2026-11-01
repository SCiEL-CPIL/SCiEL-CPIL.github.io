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
    4. ts_index: dict mapping t-subset tuple to a unique integer index
    """
    all_blocks = []
    block_to_ts = []
    ts_to_blocks = {}
    ts_index = {}
    
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
                ts_index[ts_tuple] = len(ts_index)
            ts_to_blocks[ts_tuple].append(idx)
            
    return all_blocks, block_to_ts, ts_to_blocks, ts_index

def _greedy_cover(v, k, t, all_blocks, block_to_ts, ts_to_blocks, ts_index, rng, deadline):
    """
    Greedy set cover using bitmask for speed.
    """
    n_ts = len(ts_index)
    uncovered_mask = (1 << n_ts) - 1  # All t-subsets uncovered initially
    blocks = []
    
    while uncovered_mask and time.time() < deadline:
        # Pick a random uncovered t-subset
        # Find an uncovered bit
        tmp = uncovered_mask
        seed_bit = 0
        while not (tmp & 1):
            tmp >>= 1
            seed_bit += 1
        
        # Find the t-subset corresponding to this bit
        # We need a reverse mapping from bit index to t-subset
        # For efficiency, let's just iterate through ts_index
        # Actually, let's precompute a list of t-subsets in order of their index
        # But that's expensive. Let's use a different approach.
        
        # For now, let's pick a random uncovered t-subset by iterating
        uncovered_ts_list = []
        for ts_t, idx in ts_index.items():
            if (uncovered_mask >> idx) & 1:
                uncovered_ts_list.append(ts_t)
                if len(uncovered_ts_list) >= 100:  # Limit for speed
                    break
        
        if not uncovered_ts_list:
            break
            
        seed_ts = rng.choice(uncovered_ts_list)
        
        # Get candidate blocks that cover seed_ts
        candidates = ts_to_blocks.get(seed_ts, [])
        if not candidates:
            break
            
        best_idx = -1
        best_gain = -1
        
        for idx in candidates:
            # Calculate gain: how many uncovered t-subsets does this block cover?
            gain = 0
            for ts in block_to_ts[idx]:
                ts_idx = ts_index[ts]
                if (uncovered_mask >> ts_idx) & 1:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                
        if best_idx < 0:
            break
            
        blocks.append(all_blocks[best_idx])
        
        # Remove covered t-subsets from uncovered
        for ts in block_to_ts[best_idx]:
            ts_idx = ts_index[ts]
            uncovered_mask &= ~(1 << ts_idx)
            
    return blocks

def _local_search(blocks, v, k, t, all_blocks, block_to_ts, ts_to_blocks, ts_index, deadline):
    """
    Local search:
    1. Remove redundant blocks.
    2. Try swap operations: replace one block with another that covers more.
    """
    blocks = [list(b) for b in blocks]
    
    def build_coverage(blocks):
        counts = {}
        for b in blocks:
            b_set = frozenset(b)
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
                # Remove block i
                removed = blocks.pop(i)
                for ts in itertools.combinations(removed, t):
                    ts_t = tuple(sorted(ts))
                    coverage[ts_t] -= 1
                    if coverage[ts_t] <= 0:
                        del coverage[ts_t]
                improved = True
                break # Restart loop to recalculate n and indices
        
        if improved:
            continue
            
        # Phase 2: Try swap operations
        # For each block, try replacing it with a different block that covers at least the same subsets
        n = len(blocks)
        for i in range(n):
            if time.time() >= deadline:
                break
                
            b_i = blocks[i]
            b_i_set = frozenset(b_i)
            
            # Find subsets uniquely covered by b_i
            unique_to_i = set()
            for ts in itertools.combinations(b_i, t):
                ts_t = tuple(sorted(ts))
                if coverage.get(ts_t, 0) == 1:
                    unique_to_i.add(ts_t)
            
            if not unique_to_i:
                continue  # This block is redundant, should have been removed in Phase 1
            
            # Try to find a replacement block that covers all unique_to_i subsets
            # and has fewer "wasted" subsets (subsets not in unique_to_i)
            best_j = -1
            best_score = float('inf')
            
            # Candidate blocks: those that cover all subsets in unique_to_i
            # We can find these by looking at blocks that contain the intersection of all points in unique_to_i
            # Actually, let's just iterate through all blocks and check
            
            for j, b_j in enumerate(all_blocks):
                if time.time() >= deadline:
                    break
                if j == i:
                    continue
                    
                b_j_set = frozenset(b_j)
                
                # Check if b_j covers all unique_to_i subsets
                covers_all = True
                for ts_t in unique_to_i:
                    if not b_j_set.issuperset(ts_t):
                        covers_all = False
                        break
                if not covers_all:
                    continue
                
                # Score: number of subsets in b_j that are NOT in unique_to_i
                # Lower is better
                score = 0
                for ts in itertools.combinations(b_j, t):
                    ts_t = tuple(sorted(ts))
                    if ts_t not in unique_to_i:
                        score += 1
                        
                if score < best_score:
                    best_score = score
                    best_j = j
                    
            if best_j >= 0:
                # Replace block i with all_blocks[best_j]
                old_b = blocks[i]
                for ts in itertools.combinations(old_b, t):
                    ts_t = tuple(sorted(ts))
                    coverage[ts_t] -= 1
                    if coverage[ts_t] <= 0:
                        del coverage[ts_t]
                        
                new_b = all_blocks[best_j]
                blocks[i] = list(new_b)
                for ts in itertools.combinations(new_b, t):
                    ts_t = tuple(sorted(ts))
                    coverage[ts_t] = coverage.get(ts_t, 0) + 1
                    
                improved = True
                break  # Restart to re-evaluate
                
            if time.time() >= deadline:
                break
                
    return blocks

def construct(v, k, t):
    global _global_deadline
    _global_deadline = time.time() + 145
    rng = random.Random(42)
    
    # Precompute
    all_blocks, block_to_ts, ts_to_blocks, ts_index = _precompute(v, k, t)
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple greedy restarts
    for restart in range(5):
        if time.time() > _global_deadline - 30:
            break
            
        blocks = _greedy_cover(v, k, t, all_blocks, block_to_ts, ts_to_blocks, ts_index, rng, _global_deadline - 30)
        
        if len(blocks) < best_count:
            best_blocks = [list(b) for b in blocks]
            best_count = len(blocks)
            
        # Local search on this solution
        if time.time() < _global_deadline - 10:
            improved_blocks = _local_search(blocks, v, k, t, all_blocks, block_to_ts, ts_to_blocks, ts_index, _global_deadline - 10)
            if len(improved_blocks) < best_count:
                best_blocks = improved_blocks
                best_count = len(improved_blocks)
                
    return best_blocks
