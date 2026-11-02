"""Improved greedy covering for C(19,8,5) with bitset acceleration and deadline guard."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s (or 150s) kill

    # Total number of t-subsets
    from math import comb
    n_tsub = comb(v, t)

    # Map each t-subset to an integer index
    # We'll use a dictionary for the mapping since t-subsets are tuples
    sub_to_idx = {}
    idx = 0
    for sub in itertools.combinations(range(v), t):
        sub_to_idx[sub] = idx
        idx += 1

    # Represent uncovered t-subsets as a bitset (numpy boolean array)
    uncovered = np.ones(n_tsub, dtype=bool)

    # For each point x, precompute which t-subsets contain x
    # point_subsets[x] is a list of indices of t-subsets that contain x
    point_subsets = [[] for _ in range(v)]
    for sub, i in sub_to_idx.items():
        for x in sub:
            point_subsets[x].append(i)
    
    # Convert to numpy arrays for faster operations
    point_subsets = [np.array(lst, dtype=np.int64) for lst in point_subsets]

    blocks = []
    
    while True:
        if time.time() > deadline:
            break
        if not uncovered.any():
            break
        
        # Pick the first uncovered t-subset to start a new block
        first_uncovered_idx = int(np.argmax(uncovered))
        # Find the actual subset
        # We need a reverse mapping
        # Let's build it lazily or use a different approach
        # Actually, let's just track the subset directly
        # Rebuild: iterate to find the subset for the index
        # This is slow. Let's keep a list of subsets.
        
        # Better approach: keep a list of all subsets
        all_subsets = list(itertools.combinations(range(v), t))
        # sub_to_idx is already built from this list
        
        start_sub = all_subsets[first_uncovered_idx]
        block = list(start_sub)
        
        while len(block) < k:
            if time.time() > deadline:
                break
            
            best_point = -1
            best_gain = -1
            block_set = set(block)
            
            for x in range(v):
                if x in block_set:
                    continue
                
                # Gain: number of uncovered t-subsets that contain x and are subsets of (block + {x})
                # A t-subset S is covered by adding x to block if:
                #   S contains x, and
                #   S \ {x} is a subset of block
                # Since |S| = t, S \ {x} has size t-1
                # So we need: x in S, and S - {x} ⊆ block
                
                # For efficiency: consider all (t-1)-subsets of block
                # For each (t-1)-subset R of block, the t-subset R ∪ {x} is covered
                # Check if it's uncovered
                
                gain = 0
                # Iterate over (t-1)-combinations of block
                for rest in itertools.combinations(block, t - 1):
                    s = tuple(sorted(rest + (x,)))
                    s_idx = sub_to_idx[s]
                    if uncovered[s_idx]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1:
                # Should not happen if uncovered is non-empty
                break
            block.append(best_point)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered: mark all t-subsets of this block as covered
        for sub in itertools.combinations(block, t):
            uncovered[sub_to_idx[sub]] = False
    
    return blocks
# EVOLVE-BLOCK-END
