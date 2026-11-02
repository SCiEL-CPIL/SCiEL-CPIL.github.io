import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generate all vectors of norm sqrt(3)
    # 3 non-zero coordinates, each +/- 1
    vectors = []
    for indices in itertools.combinations(range(D), 3):
        i, j, k = indices
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    vectors.append(np.array(v, dtype=np.int64))
    
    n_vectors = len(vectors)
    # Pre-compute dot products? 
    # Matrix of size 1320x1320 is ~1.7M entries. 
    # 1.7M integers is ~13MB. 
    # Computing it takes 1320*1320 dot products of length 11.
    # 1.7M * 11 = 19M ops. Fast.
    
    # However, we only need to check compatibility with selected ones.
    # Greedy with random restarts.
    
    best_set = []
    best_size = 0
    
    # Try to improve upon the seed (220)
    # If we find a set > 220, we return it.
    
    num_restarts = 0
    while time.time() < deadline:
        # Random shuffle
        idx = np.arange(n_vectors)
        np.random.shuffle(idx)
        
        selected = []
        selected_indices = set()
        
        for i in idx:
            v = vectors[i]
            # Check compatibility with all selected
            compatible = True
            for j in selected_indices:
                # Dot product must be <= 1
                # v . vectors[j] <= 1
                if np.dot(v, vectors[j]) > 1:
                    compatible = False
                    break
            
            if compatible:
                selected.append(v)
                selected_indices.add(i)
                
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
            
        num_restarts += 1
        # If we get stuck at a local optimum, maybe try a different strategy?
        # For now, simple random greedy.
        
        # Optimization: if best_size > 220, we can stop early?
        # No, we want the max. But if we find > 220, it's already an improvement.
        # We can break if we find a very high number, but let's run until deadline.
        
    if best_size > 220:
        return np.array(best_set, dtype=np.int64)
    else:
        # Fallback to seed if greedy fails to beat 220
        vectors_seed = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors_seed.append(v)
        return np.array(vectors_seed, dtype=np.int64)
