import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = 140.0
    
    # Generate all integer vectors in R^11 with norm squared 4
    # Types:
    # 1. (±2, 0, ..., 0) : 22 vectors
    # 2. (±1, ±1, 0, ..., 0) : 220 vectors
    # 3. (±1, ±1, ±1, ±1, 0, ..., 0) : 5280 vectors
    
    vectors = []
    
    # Type 1
    for i in range(D):
        for s in (1, -1):
            v = [0]*D
            v[i] = 2*s
            vectors.append(v)
            
    # Type 2
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0]*D
                v[i] = si
                v[j] = sj
                vectors.append(v)
                
    # Type 3
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            v = [0]*D
            for idx, s in zip(combo, signs):
                v[idx] = s
            vectors.append(v)
            
    V = np.array(vectors, dtype=np.int64)
    # V shape (5522, 11)
    
    # We want to select 512 vectors such that min pairwise dist >= 2
    # dist^2 >= 4
    # Since norm^2 = 4 for all, dist^2 = 4 + 4 - 2<u,v> = 8 - 2<u,v>
    # dist^2 >= 4 => 8 - 2<u,v> >= 4 => 2<u,v> <= 4 => <u,v> <= 2
    # Also <u,v> is even integer.
    # So <u,v> can be 0, -2, -4, ...
    # If <u,v> = 2, dist^2 = 4. OK.
    # If <u,v> = 4, dist^2 = 0. Bad.
    # So we need <u,v> <= 2 for all pairs.
    
    # Greedy selection
    selected_indices = []
    # Sort by norm (all same) or just take in order
    # To be fast, we can try to pick the "most orthogonal" ones first?
    # Or just greedy.
    
    # Precompute dot products? 5522x5522 is ~30M entries. 30M * 8 bytes = 240MB. OK.
    # But calculating it takes time.
    # Let's try a greedy approach with incremental checks.
    
    # We know the 242 vectors of Type 1 and 2 are mutually compatible?
    # Check Type 1 vs Type 1: <u,v> = 4 if same, 0 if diff.
    # Check Type 1 vs Type 2: <u,v> = 0 or ±2.
    # Check Type 2 vs Type 2: <u,v> can be 2 (if share one index with same sign), 0, -2, 4 (if same).
    # So the 242 vectors form a valid set?
    # Max <u,v> is 2. So dist^2 >= 4. Yes.
    
    selected = V[:242]
    
    # Now add Type 3 vectors (indices 242 to end)
    # Check compatibility with existing selected.
    # For a new vector v, check max dot product with selected.
    # If max <v, u> <= 2, add it.
    
    for i in range(242, len(V)):
        if time.time() > deadline:
            break
        v = V[i]
        # Compute dot products with selected
        # selected is (k, 11)
        dots = selected @ v
        if np.max(dots) <= 2:
            selected = np.vstack([selected, v])
            if len(selected) == 512:
                break
                
    return selected

