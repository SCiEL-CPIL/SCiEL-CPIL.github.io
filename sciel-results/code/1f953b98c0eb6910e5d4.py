import numpy as np
import itertools
import time
import random

D = 11

def construct():
    deadline = time.time() + 115  # Safety margin
    
    # Generate all 4-subsets of range(11)
    subsets = list(itertools.combinations(range(D), 4))
    n = len(subsets)  # 330
    
    # Convert subsets to bitmasks for fast intersection checks
    masks = []
    for s in subsets:
        m = 0
        for pos in s:
            m |= (1 << pos)
        masks.append(m)
        
    # Precompute adjacency list
    # i and j conflict if they share exactly 3 elements
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            inter = (masks[i] & masks[j])
            # Count bits
            # Python 3.10+ has int.bit_count()
            # Assuming modern Python, otherwise use bin(x).count("1")
            if inter.bit_count() == 3:
                adj[i].add(j)
                adj[j].add(i)
                
    def greedy_mis(seed_val):
        rng = random.Random(seed_val)
        order = list(range(n))
        rng.shuffle(order)
        is_set = set()
        for v in order:
            # Check if v has any neighbor in is_set
            # Optimization: if we maintained a blocked set, it would be O(1)
            # But with N=330, O(Degree) is fine.
            if not (adj[v] & is_set):
                is_set.add(v)
        return is_set

    def local_search(is_set, rng):
        is_set = set(is_set)
        # Try to add any node that doesn't conflict
        # Repeat until no improvement
        while True:
            improved = False
            # Get candidates not in is_set
            candidates = [v for v in range(n) if v not in is_set]
            rng.shuffle(candidates)
            for v in candidates:
                if not (adj[v] & is_set):
                    is_set.add(v)
                    improved = True
            if not improved:
                break
        return is_set

    best_is = set()
    best_size = 0
    
    # Run multiple restarts
    # Estimate time per restart:
    # Greedy: 330 iterations * set intersection (size ~28). Very fast.
    # Local search: Few iterations.
    # We can run many restarts.
    
    current_seed = 0
    while time.time() < deadline:
        is_set = greedy_mis(current_seed)
        is_set = local_search(is_set, random.Random(current_seed))
        
        if len(is_set) > best_size:
            best_size = len(is_set)
            best_is = is_set
            
        current_seed += 1
        
        # If we found a very large set (e.g., > 50), we might stop early if time is tight?
        # No, keep searching to ensure we find the max.
        
    # Build vectors
    vectors = []
    for idx in best_is:
        s = subsets[idx]
        # For each selected subset, add all 16 sign variations
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            vectors.append(v)
    
    return np.array(vectors, dtype=np.int64)
