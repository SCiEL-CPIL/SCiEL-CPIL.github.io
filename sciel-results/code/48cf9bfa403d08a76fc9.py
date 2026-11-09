"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using multi-start greedy with efficient reduction.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    N_t = len(all_t_subsets)
    subset_to_id = {s: i for i, s in enumerate(all_t_subsets)}
    
    def greedy_cover(seed=None):
        """Run one greedy pass. Returns blocks list."""
        if seed is not None:
            random.seed(seed)
        
        # Track uncovered t-subsets as a set of IDs
        uncovered = set(range(N_t))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset to start a new block
            start_id = random.choice(list(uncovered))
            start_subset = all_t_subsets[start_id]
            block = list(start_subset)
            block_set = set(block)
            
            while len(block) < k:
                # Find point that maximizes coverage of uncovered t-subsets
                best_gain = -1
                best_point = None
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        cand = tuple(sorted(rest + (x,)))
                        cid = subset_to_id.get(cand)
                        if cid is not None and cid in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Remove all t-subsets covered by this block
            for comb in itertools.combinations(block, t):
                cid = subset_to_id.get(comb)
                if cid is not None:
                    uncovered.discard(cid)
        
        return blocks
    
    def reduce_blocks(blocks):
        """Try to remove redundant blocks from the covering using coverage counts."""
        if not blocks:
            return blocks
        
        # Build coverage count: for each t-subset ID, count how many blocks cover it
        coverage_count = np.zeros(N_t, dtype=np.int32)
        block_to_tsubs = []  # For each block, list of t-subset IDs it covers
        
        for b in blocks:
            tsubs = []
            for comb in itertools.combinations(b, t):
                cid = subset_to_id.get(comb)
                if cid is not None:
                    tsubs.append(cid)
                    coverage_count[cid] += 1
            block_to_tsubs.append(tsubs)
        
        reduced = list(blocks)
        reduced_tsubs = list(block_to_tsubs)
        changed = True
        
        while changed:
            changed = False
            for i in range(len(reduced)):
                # Check if block i is redundant (all its t-subsets are covered by others)
                is_redundant = True
                for cid in reduced_tsubs[i]:
                    if coverage_count[cid] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove block i
                    for cid in reduced_tsubs[i]:
                        coverage_count[cid] -= 1
                    reduced.pop(i)
                    reduced_tsubs.pop(i)
                    changed = True
                    break
        
        return reduced
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple restarts
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        seed = restart_count * 7919  # Use a prime for better distribution
        blocks = greedy_cover(seed=seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
        
        # Try reduction
        if time.time() < deadline - 5.0:
            reduced = reduce_blocks(best_blocks)
            if len(reduced) < best_count:
                best_count = len(reduced)
                best_blocks = reduced
    
    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered = set(range(N_t))
        blocks = []
        while uncovered:
            start_id = min(uncovered)
            start_subset = all_t_subsets[start_id]
            block = list(start_subset)
            block_set = set(block)
            
            while len(block) < k:
                best_gain = -1
                best_point = None
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        cand = tuple(sorted(rest + (x,)))
                        cid = subset_to_id.get(cand)
                        if cid is not None and cid in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            for comb in itertools.combinations(block, t):
                cid = subset_to_id.get(comb)
                if cid is not None:
                    uncovered.discard(cid)
        
        best_blocks = blocks
    
    return best_blocks
# EVOLVE-BLOCK-END
