import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = 145  # Safety margin
    
    # 1. Generate candidate vectors
    # Type A: Norm 2, 2 non-zeros (+/- 2 on one axis? No, +/- 2 e_i is norm 2. +/- 1 on two axes is norm sqrt(2).)
    # Wait, the task says "integer vectors".
    # Norm of (2, 0, ...) is 2.
    # Norm of (1, 1, ...) is sqrt(2).
    # If we mix, max norm is 2.
    # Distances must be >= 2.
    # We analyzed that Type A (2,0,..) and Type B (1,1,1,1,..) are compatible.
    # What about Type C (1,1,..) norm sqrt(2)?
    # If we include Type C, max norm is still 2 (if we have Type A/B).
    # Dist between Type C and Type A:
    # u=(1,1,0..), v=(2,0,0..).
    # u.v = 2.
    # ||u-v||^2 = 2 + 4 - 4 = 2. Dist sqrt(2) < 2.
    # So Type C is NOT compatible with Type A.
    # So we should probably stick to Norm 2 vectors (Type A and Type B).
    
    vectors = []
    # Type A: +/- 2 e_i
    for i in range(D):
        v = [0]*D
        v[i] = 2
        vectors.append(v)
        v[i] = -2
        vectors.append(v)
        
    # Type B: +/- 1 on 4 axes
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0]*D
            for idx, s in zip(support, signs):
                v[idx] = s
            vectors.append(v)
            
    arr = np.array(vectors, dtype=np.int64)
    n = len(arr)
    
    # 2. Build Conflict List
    # Conflict if dot product > 2.
    # We can compute this efficiently.
    
    # To avoid O(n^2) full matrix, we can group by support.
    # But for 5300 nodes, even O(n^2) with numpy might be okay if vectorized.
    # 5300x5300 dot product matrix.
    
    # Let's try to build a sparse conflict list.
    # For each vector, find conflicts.
    
    # Optimization: 
    # Most conflicts are between Type B vectors with overlap 3.
    # Type A vectors have no conflicts with Type B?
    # We checked: Type A (2,0,..) and Type B (1,1,1,1,..).
    # If overlap 1: dot prod 2. Dist 2. OK.
    # If overlap 0: dot prod 0. Dist > 2. OK.
    # So Type A is compatible with ALL Type B.
    # Type A vs Type A:
    # (2,0,..) and (2,0,..) -> same.
    # (2,0,..) and (-2,0,..) -> dist 4. OK.
    # (2,0,..) and (2,0,..) on diff axis -> dist sqrt(8). OK.
    # So ALL Type A vectors are mutually compatible and compatible with all Type B.
    
    # So we can always include all 22 Type A vectors.
    # We just need to find a large independent set among the 5280 Type B vectors.
    
    type_a_count = 22
    type_b_vectors = arr[type_a_count:]
    n_b = len(type_b_vectors)
    
    # Find independent set in Type B graph.
    # Conflicts only if overlap 3 and signs match on overlap.
    
    # Heuristic: Greedy with random restarts.
    
    best_set = set()
    
    # To speed up conflict checking, we can precompute conflicts for Type B.
    # But 5280 nodes is small enough for a few iterations of greedy.
    
    # Let's use a bitset representation for the current set to check compatibility fast?
    # Or just a list and a dictionary of conflicts.
    
    # Build conflict dictionary for Type B
    # conflicts[i] = set of j such that i, j conflict.
    
    # This might be slow to build.
    # Alternative: Check compatibility on the fly.
    
    def is_compatible(v1, current_set_vecs):
        # v1 is numpy array
        # current_set_vecs is numpy array of shape (k, D)
        if len(current_set_vecs) == 0:
            return True
        dots = np.dot(current_set_vecs, v1)
        return np.max(dots) <= 2
        
    # Greedy
    def greedy_search(vecs, n_iter=10):
        best_size = 0
        best_indices = []
        
        for _ in range(n_iter):
            if time.time() > deadline - 5:
                break
                
            order = np.random.permutation(len(vecs))
            selected = []
            selected_vecs = []
            
            for idx in order:
                v = vecs[idx]
                # Check against selected
                if not selected_vecs:
                    selected.append(idx)
                    selected_vecs.append(v)
                    continue
                
                # Vectorized check
                dots = np.dot(np.array(selected_vecs), v)
                if np.max(dots) <= 2:
                    selected.append(idx)
                    selected_vecs.append(v)
                    
            if len(selected) > best_size:
                best_size = len(selected)
                best_indices = selected
                
        return best_indices

    # Run greedy search on Type B
    # 5280 vectors.
    # Each iteration O(N^2 * D). 5280^2 * 11 ~ 300 million ops.
    # In Python, this is too slow for many iterations.
    # We need to optimize.
    
    # Precompute a "compatibility" check using bitsets or similar?
    # Or reduce the search space.
    
    # Let's try a different heuristic:
    # Group by support.
    # There are 330 supports.
    # For each support, we have 16 vectors.
    # We can decide which subset to pick.
    
    # Let's just return the 220 vectors from the seed program if search fails,
    # but we know we can do better.
    
    # Actually, the seed program returns 220.
    # If we just return the 220, score is 1.0.
    # We need ~580.
    
    # Let's try to construct a set using the "1 pattern per support" idea first.
    # Pick 1 sign pattern for each of the 330 supports.
    # e.g., all +1.
    # Then we have 330 vectors.
    # Check if they are mutually compatible.
    # Conflicts if overlap 3.
    # We can remove conflicting ones.
    
    # Let's implement a simple local search on the 330 supports.
    # State: For each support, a chosen sign pattern (or None).
    # Objective: Maximize number of chosen patterns.
    
    # This is complex to code quickly.
    
    # Let's go with the 220 vectors (seed) + 22 Type A?
    # Seed vectors are Type C (norm sqrt(2)).
    # Type A is norm 2.
    # Incompatible.
    
    # So we must choose between Type C (220 vectors) and Type A+B.
    # Type C gives 220.
    # Type A+B can give up to 5300.
    
    # Let's try to find a large independent set in Type B using a fast C-based library?
    # No, only standard library.
    
    # Let's try a very fast greedy:
    # Randomly shuffle Type B vectors.
    # Add to set if compatible.
    # To check compatibility, we maintain a list of selected vectors.
    # To speed up, we can use numpy.
    
    # 5280 vectors.
    # If we do 10 restarts, and each restart takes 1s, that's 10s.
    
    # Let's try it.
    
    indices = greedy_search(type_b_vectors, n_iter=20)
    
    if indices:
        selected_b = type_b_vectors[indices]
        # Add all Type A
        selected_a = arr[:type_a_count]
        result = np.vstack([selected_a, selected_b])
    else:
        # Fallback to seed
        result = arr[:220] # This is not correct, arr is A+B.
        # Let's just return the 220 Type C vectors if we fail?
        # We didn't generate Type C in this block.
        
    # Wait, I need to generate Type C for fallback.
    
    # Let's assume the greedy finds a good set.
    
    return result

# The code above has issues with fallback and generation.
# I will rewrite it cleanly.
