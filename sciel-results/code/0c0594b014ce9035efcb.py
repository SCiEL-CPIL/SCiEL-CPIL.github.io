import numpy as np
import itertools

D = 11

def construct():
    # Strategy: Use vectors of norm squared 4.
    # 1. Axis vectors: (±2, 0, ..., 0). Count = 22.
    # 2. Four-nonzero vectors: (±1, ±1, ±1, ±1, 0, ...). Count = 5280.
    # Condition for norm squared 4 vectors: dot product <= 2.
    # Axis vectors are compatible with all four-nonzero vectors (max dot 2).
    # We need to select a large subset of four-nonzero vectors with pairwise dot <= 2.
    
    # Generate axis vectors
    axis_vectors = []
    for i in range(D):
        for s in (2, -2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            axis_vectors.append(v)
            
    # Generate four-nonzero vectors
    four_nonzero = []
    indices = range(D)
    for combo in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, sign in zip(combo, signs):
                v[idx] = sign
            four_nonzero.append(v)
            
    four_nonzero = np.array(four_nonzero)
    
    # Greedy selection of four-nonzero vectors
    # We want max subset with pairwise dot <= 2.
    # Randomize order to escape local minima, try a few times.
    best_selected = []
    np.random.seed(42)
    
    for _ in range(5): # Try 5 times
        order = np.random.permutation(len(four_nonzero))
        selected_idx = []
        # To speed up, we can maintain a list of selected vectors and check new ones against them.
        # With 5000 vectors, O(N^2) is 25M ops. In Python loop, this is slow.
        # We can vectorize the check against the *currently selected* set.
        # But selected set grows.
        
        # Alternative: Simple greedy.
        # Given time limit 120s, and 5280 vectors, a pure Python loop might be too slow if we do 5280 * 5280 checks.
        # 27M checks in Python is ~10-30 seconds.
        # Let's try to optimize or just do one pass.
        
        selected_vecs = []
        for i in order:
            v = four_nonzero[i]
            # Check against all selected
            if not selected_vecs:
                selected_vecs.append(v)
                selected_idx.append(i)
                continue
                
            # Compute dot products
            # selected_vecs is a list of arrays. Convert to array for batch?
            # Or just loop.
            ok = True
            for sv in selected_vecs:
                if np.dot(v, sv) > 2:
                    ok = False
                    break
            if ok:
                selected_vecs.append(v)
                selected_idx.append(i)
                
        if len(selected_idx) > len(best_selected):
            best_selected = selected_idx
            
    # Combine axis vectors and selected four-nonzero vectors
    all_vectors = axis_vectors + [four_nonzero[i] for i in best_selected]
    
    return np.array(all_vectors)
