"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from math import comb


def construct(v, k, t):
    deadline = time.time() + 115.0  # safety margin before 120s limit
    
    all_points = list(range(v))
    all_tsubs = list(itertools.combinations(all_points, t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {s: i for i, s in enumerate(all_tsubs)}
    
    # Track uncovered t-subsets as a bitset for efficiency
    uncovered_mask = [(1 << i) for i in range(n_tsubs)]  # list of single-bit masks
    # Use a set of indices for uncovered
    uncovered = set(range(n_tsubs))
    
    # Precompute: for each point x, which t-subset indices contain x
    point_to_tsubs = [[] for _ in range(v)]
    for idx, s in enumerate(all_tsubs):
        for x in s:
            point_to_tsubs[x].append(idx)
    
    # Precompute: for each (t-1)-subset, the list of t-subset indices it extends to
    # i.e., for a (t-1)-subset R, which t-subsets contain R?
    # Actually, we need: given a block B (size < k) and candidate x, 
    # gain = number of uncovered t-subsets that contain x and (t-1) elements from B
    # = sum over (t-1)-subsets R of B: count of uncovered t-subsets containing R ∪ {x}
    
    # For efficiency, precompute for each (t-1)-subset, the t-subsets that contain it
    # But that's too many. Instead, compute on the fly but cache.
    
    blocks = []
    
    while uncovered and time.time() < deadline:
        # Pick a seed t-subset: the one with minimum index (deterministic)
        seed_idx = min(uncovered)
        seed = all_tsubs[seed_idx]
        block_set = set(seed)
        block = list(seed)
        
        while len(block) < k:
            if time.time() > deadline:
                break
            
            best_x = None
            best_gain = -1
            
            for x in all_points:
                if x in block_set:
                    continue
                
                # Calculate gain: number of uncovered t-subsets that would be covered
                # by adding x to current block
                gain = 0
                # A t-subset is covered if it contains x and (t-1) elements from block
                for rest in itertools.combinations(block, t - 1):
                    # Form the t-subset: rest + (x,)
                    ts = tuple(sorted(rest + (x,)))
                    idx = tsub_to_idx.get(ts, -1)
                    if idx >= 0 and idx in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None or best_gain <= 0:
                # No point improves coverage; add arbitrary point to fill
                for x in all_points:
                    if x not in block_set:
                        best_x = x
                        break
                if best_x is None:
                    break
            
            block.append(best_x)
            block_set.add(best_x)
        
        if len(block) < k:
            break
        
        block.sort()
        
        # Check for duplicate
        block_tuple = tuple(block)
        if block_tuple not in [tuple(b) for b in blocks]:
            blocks.append(block)
        
        # Update uncovered: remove all t-subsets of this block
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx.get(ts, -1)
            if idx >= 0:
                uncovered.discard(idx)
        
        if not uncovered:
            break
    
    # If we ran out of time and still have uncovered, do a fast completion
    # Just greedily cover remaining without optimization
    while uncovered:
        seed_idx = min(uncovered)
        seed = all_tsubs[seed_idx]
        block_set = set(seed)
        block = list(seed)
        
        while len(block) < k:
            # Add arbitrary point not in block
            for x in all_points:
                if x not in block_set:
                    block.append(x)
                    block_set.add(x)
                    break
            else:
                break
        
        if len(block) < k:
            break
        
        block.sort()
        blocks.append(block)
        
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx.get(ts, -1)
            if idx >= 0:
                uncovered.discard(idx)
        
        if not uncovered:
            break
    
    return blocks
# EVOLVE-BLOCK-END
