"""Covering design C(19,8,5) - optimized greedy with redundancy removal."""
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Coverage count array: how many blocks cover each t-subset
    coverage = np.zeros(n_tsubsets, dtype=np.int32)
    
    # For efficient gain computation, precompute for each point x and each 
    # (t-1)-subset of points, the list of t-subset indices containing x and that (t-1)-subset
    # Actually, let's compute gain on the fly using the tsubset_to_idx dict
    
    def compute_gain(partial_block, candidate_x):
        """Count uncovered t-subsets that would be covered by adding candidate_x to partial_block."""
        if candidate_x in partial_block:
            return 0
        count = 0
        partial_set = set(partial_block)
        for rest in itertools.combinations(partial_block, t - 1):
            ts = tuple(sorted(rest + (candidate_x,)))
            idx = tsubset_to_idx.get(ts)
            if idx is not None and coverage[idx] == 0:
                count += 1
        return count
    
    def add_block(block):
        """Update coverage for all t-subsets in the block."""
        for ts in itertools.combinations(block, t):
            idx = tsubset_to_idx[ts]
            coverage[idx] += 1
    
    def remove_block(block):
        """Decrease coverage for all t-subsets in the block."""
        for ts in itertools.combinations(block, t):
            idx = tsubset_to_idx[ts]
            coverage[idx] -= 1
    
    blocks = []
    
    # Greedy construction
    while True:
        if time.time() > deadline - 30.0:
            break
        if np.all(coverage > 0):
            break
        
        # Find an uncovered t-subset to start with
        uncovered_indices = np.where(coverage == 0)[0]
        if len(uncovered_indices) == 0:
            break
        # Pick the first uncovered t-subset (or randomize for diversity)
        start_idx = uncovered_indices[0]
        start_subset = all_tsubsets[start_idx]
        
        block = list(start_subset)
        
        # Add points greedily
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                g = compute_gain(block, x)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        add_block(block)
    
    # Redundancy removal pass
    def redundancy_removal():
        """Remove blocks that don't uniquely cover any t-subset."""
        nonlocal blocks, coverage
        changed = True
        while changed and time.time() < deadline - 10.0:
            changed = False
            i = 0
            while i < len(blocks):
                if time.time() > deadline - 10.0:
                    break
                # Check if block i is redundant
                block = blocks[i]
                is_redundant = True
                for ts in itertools.combinations(block, t):
                    idx = tsubset_to_idx[ts]
                    if coverage[idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    remove_block(block)
                    blocks.pop(i)
                    changed = True
                else:
                    i += 1
    
    redundancy_removal()
    
    # If we have time, try a second greedy pass to fill any remaining gaps
    # (shouldn't be needed if first pass completed, but just in case)
    while time.time() < deadline - 5.0:
        if np.all(coverage > 0):
            break
        uncovered_indices = np.where(coverage == 0)[0]
        if len(uncovered_indices) == 0:
            break
        start_idx = uncovered_indices[0]
        start_subset = all_tsubsets[start_idx]
        block = list(start_subset)
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                g = compute_gain(block, x)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
        block.sort()
        blocks.append(block)
        add_block(block)
    
    # Final redundancy removal
    if time.time() < deadline - 2.0:
        redundancy_removal()
    
    return blocks
