import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def _get_offsets(n, max_dist):
    """Generate all binary vectors of length n with weight <= max_dist."""
    offsets = [0]
    for k in range(1, max_dist + 1):
        # Use itertools.combinations to generate indices
        from itertools import combinations
        for comb in combinations(range(n), k):
            mask = 0
            for i in comb:
                mask |= (1 << i)
            offsets.append(mask)
    return np.array(offsets, dtype=np.uint32)

def construct():
    deadline = time.time() + 55.0
    N = 17
    MAX_DIST_FORBIDDEN = 5 # Mark neighbors up to distance 5
    
    # Precompute offsets
    offsets = _get_offsets(N, MAX_DIST_FORBIDDEN)
    universe_size = 1 << N
    
    best_code = []
    
    # Define a greedy search function
    def greedy_search(order):
        forbidden = np.zeros(universe_size, dtype=bool)
        code = []
        for i in order:
            if not forbidden[i]:
                code.append(i)
                # Mark neighbors
                # i is int, offsets is uint32 array
                # i ^ offsets works if i is compatible
                # Need to ensure i is treated as uint32 or compatible
                # numpy handles int ^ uint32 array by casting
                forbidden[i ^ offsets] = True
        return code

    # Strategy 1: Natural order
    if time.time() < deadline:
        order = np.arange(universe_size, dtype=np.uint32)
        code = greedy_search(order)
        if len(code) > len(best_code):
            best_code = code

    # Strategy 2: Random shuffles
    rng = random.Random(42)
    indices = np.arange(universe_size, dtype=np.uint32)
    
    # Try a few random shuffles
    while time.time() < deadline - 5.0: # Keep 5s margin
        rng.shuffle(indices)
        code = greedy_search(indices)
        if len(code) > len(best_code):
            best_code = code
            
    return best_code

# EVOLVE-BLOCK-END
