"""Covering design C(19,7,4) using vectorized greedy + local optimization."""
import time
import numpy as np
from itertools import combinations
from math import comb


def _all_tsubsets(v, t):
    """Generate all t-subsets as a list of tuples."""
    return list(combinations(range(v), t))


def _subset_to_idx(v, t):
    """Map a t-subset to a unique index using combinatorial number system."""
    def comb_index(subset, v, t):
        # Use the rank of the combination in lex order
        idx = 0
        r = t
        for i in range(t):
            # For each element in the combination, add C(element - i, t - i)
            # Actually, use the standard combinatorial number system
            pass
        # Simpler: use a dictionary cache
        return None
    
    # For small v,t, just use a dict
    cache = {}
    idx = 0
    for sub in combinations(range(v), t):
        cache[sub] = idx
        idx += 1
    return cache


def _build_index(v, t):
    """Build a mapping from t-subset tuples to indices."""
    subs = list(combinations(range(v), t))
    idx_map = {sub: i for i, sub in enumerate(subs)}
    return subs, idx_map


def _block_to_mask(block, v, t, idx_map):
    """Return a boolean mask of which t-subsets are covered by this block."""
    n_tsubs = comb(v, t)
    mask = np.zeros(n_tsubs, dtype=np.uint8)
    for sub in combinations(block, t):
        idx = idx_map.get(tuple(sub))
        if idx is not None:
            mask[idx] = 1
    return mask


def _greedy_construct(v, k, t, time_limit=110.0):
    """Vectorized greedy covering design construction."""
    start_time = time.time()
    
    # Build index for all t-subsets
    all_tsubs, idx_map = _build_index(v, t)
    n_tsubs = len(all_tsubs)
    
    # Track uncovered subsets
    uncovered = np.ones(n_tsubs, dtype=np.uint8)
    
    # Precompute for each point, which t-subsets contain it
    # point_to_tsubs[p] = list of indices of t-subsets containing p
    point_to_tsubs = [np.zeros(n_tsubs, dtype=np.uint8) for _ in range(v)]
    for i, sub in enumerate(all_tsubs):
        for p in sub:
            point_to_tsubs[p][i] = 1
    
    blocks = []
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        n_uncovered = int(np.sum(uncovered))
        if n_uncovered == 0:
            break
        
        # Greedy: pick the first uncovered t-subset to start a block
        first_uncovered_idx = int(np.argmax(uncovered))
        start_subset = all_tsubs[first_uncovered_idx]
        block = list(start_subset)
        
        # Extend block greedily
        while len(block) < k:
            if time.time() - start_time > time_limit:
                break
            
            # For each candidate point, compute how many new t-subsets it covers
            best_point = -1
            best_gain = -1
            
            for p in range(v):
                if p in block:
                    continue
                
                # Compute gain: number of uncovered t-subsets that would be covered
                # if we add p to the block
                # A t-subset is covered by block+{p} if it contains p and 
                # (t-1) elements from block
                gain = 0
                for sub in combinations(block, t - 1):
                    full_sub = tuple(sorted(sub + (p,)))
                    idx = idx_map.get(full_sub)
                    if idx is not None and uncovered[idx]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = p
            
            if best_point == -1 or best_gain <= 0:
                # No gain, just add an arbitrary point
                for p in range(v):
                    if p not in block:
                        best_point = p
                        break
            
            block.append(best_point)
        
        block.sort()
        
        # Add this block and update uncovered
        block_mask = _block_to_mask(block, v, t, idx_map)
        uncovered = uncovered & (1 - block_mask)
        blocks.append(block)
    
    return blocks


def _remove_redundant_blocks(blocks, v, t, time_limit=110.0):
    """Try to remove redundant blocks from the covering."""
    start_time = time.time()
    all_tsubs, idx_map = _build_index(v, t)
    n_tsubs = len(all_tsubs)
    
    # Build coverage matrix: rows = blocks, cols = t-subsets
    n_blocks = len(blocks)
    coverage = np.zeros((n_blocks, n_tsubs), dtype=np.uint8)
    for i, block in enumerate(blocks):
        coverage[i] = _block_to_mask(block, v, t, idx_map)
    
    # Count how many blocks cover each t-subset
    cover_count = np.sum(coverage, axis=0)
    
    removed = [False] * n_blocks
    improved = True
    
    while improved and (time.time() - start_time < time_limit):
        improved = False
        for i in range(n_blocks):
            if removed[i]:
                continue
            # Check if block i is redundant: every t-subset it covers is also covered by another block
            # i.e., for each column j where coverage[i,j] == 1, there exists i' != i with coverage[i',j] == 1
            if np.all(cover_count[coverage[i] == 1] >= 2):
                # Block i is redundant
                removed[i] = True
                cover_count -= coverage[i]
                improved = True
    
    return [blocks[i] for i in range(n_blocks) if not removed[i]]


def construct(v, k, t):
    """Construct a covering design C(v,k,t)."""
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin before 120s limit
    
    # Phase 1: Greedy construction
    blocks = _greedy_construct(v, k, t, time_limit=60.0)
    
    # Phase 2: Remove redundant blocks
    if time.time() < deadline - 10.0:
        blocks = _remove_redundant_blocks(blocks, v, t, time_limit=deadline - 10.0)
    
    return blocks
