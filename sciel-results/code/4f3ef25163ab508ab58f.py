import time
import itertools
from collections import Counter
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = 110.0  # safety margin before 120s limit
    
    # Precompute all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    
    # Map each t-subset to an index for fast lookup
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # uncovered[i] = True if subset i is not yet covered
    uncovered = bytearray([1] * n_subsets)
    uncovered_count = n_subsets
    
    blocks = []
    
    # For efficiency, precompute for each point x and each subset of size t-1 from current block,
    # which t-subsets would be covered if x is added
    # We'll compute gain on the fly but more efficiently
    
    def compute_gain(block, x):
        """Compute how many currently uncovered t-subsets would be covered by adding x to block."""
        gain = 0
        for rest in itertools.combinations(block, t - 1):
            s = tuple(sorted(rest + (x,)))
            idx = subset_to_idx.get(s)
            if idx is not None and uncovered[idx]:
                gain += 1
        return gain
    
    while uncovered_count > 0:
        if time.time() > deadline:
            break
        
        # Find an uncovered t-subset to start a new block
        # Use the first uncovered subset for determinism, or pick strategically
        # Start with the subset that appears in the fewest potential blocks (harder to cover)
        # For speed, just find any uncovered subset
        first_uncovered_idx = -1
        for i in range(n_subsets):
            if uncovered[i]:
                first_uncovered_idx = i
                break
        
        if first_uncovered_idx == -1:
            break
        
        # Start block with elements from this uncovered subset
        seed_subset = all_subsets[first_uncovered_idx]
        block = list(seed_subset[:min(t, k)])
        
        # Greedily add points to the block
        while len(block) < k:
            if time.time() > deadline:
                break
            
            # Find the point that gives the maximum gain
            best_x = -1
            best_gain = -1
            
            for x in range(v):
                if x in block:
                    continue
                g = compute_gain(block, x)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            
            if best_x == -1 or best_gain <= 0:
                # No point helps; break out to avoid infinite loop
                break
            
            block.append(best_x)
        
        if len(block) < t:
            # Can't form a valid block; skip
            break
        
        block.sort()
        blocks.append(tuple(block))
        
        # Mark all t-subsets of this block as covered
        for s in itertools.combinations(block, t):
            idx = subset_to_idx.get(s)
            if idx is not None and uncovered[idx]:
                uncovered[idx] = 0
                uncovered_count -= 1
    
    # If we ran out of time, try to ensure we have a valid covering
    # Check if all t-subsets are covered
    if uncovered_count > 0:
        # Fallback: add blocks to cover remaining subsets
        for i in range(n_subsets):
            if uncovered[i]:
                if time.time() > deadline:
                    break
                subset = all_subsets[i]
                # Extend this subset to a k-subset
                remaining_points = [x for x in range(v) if x not in subset]
                while len(subset) < k and remaining_points:
                    subset.append(remaining_points.pop(0))
                if len(subset) >= t:
                    subset = tuple(sorted(subset[:k]))
                    blocks.append(subset)
                    for s in itertools.combinations(subset, t):
                        idx = subset_to_idx.get(s)
                        if idx is not None and uncovered[idx]:
                            uncovered[idx] = 0
                            uncovered_count -= 1
    
    # Convert back to lists for the required interface
    return [list(b) for b in blocks]
