import itertools
import numpy as np
import time
import random

D = 11

def construct():
    start_time = time.time()
    DEADLINE = 115
    
    def elapsed_ok():
        return (time.time() - start_time) < DEADLINE

    # 1. Generate all Type 1 vectors
    type1_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                type1_vectors.append(v)
    
    # 2. Generate all Type 2 and Type 3 vectors
    candidates = []
    
    # Type 2: 8 ones
    for pos_set in itertools.combinations(range(D), 8):
        for mask in range(256):
            v = [0] * D
            for idx, pos in enumerate(pos_set):
                if mask & (1 << idx):
                    v[pos] = -1
                else:
                    v[pos] = 1
            candidates.append(v)
            
    # Type 3: one 2, four 1s
    for pos_2 in range(D):
        for pos_set_1 in itertools.combinations([p for p in range(D) if p != pos_2], 4):
            for s2 in (1, -1):
                for mask in range(16):
                    v = [0] * D
                    v[pos_2] = 2 * s2
                    for idx, pos in enumerate(pos_set_1):
                        if mask & (1 << idx):
                            v[pos] = -1
                        else:
                            v[pos] = 1
                    candidates.append(v)

    best_vectors = type1_vectors[:]
    best_count = len(type1_vectors)
    
    # Multiple greedy runs
    run_id = 0
    while elapsed_ok():
        run_id += 1
        random.seed(run_id)
        random.shuffle(candidates)
        
        active_vectors = []
        # We can start with a few random candidates to "break" symmetry?
        # No, starting empty is fine.
        
        # Optimization: Convert candidates to numpy array for faster iteration?
        # Iterating over list of lists is slow.
        # Pre-compute numpy array of candidates?
        # candidates_np = np.array(candidates)
        # Then iterate indices.
        
        # Let's just use the list. The bottleneck is the dot product.
        
        for v in candidates:
            if not elapsed_ok():
                break
            
            v_arr = np.array(v, dtype=np.int64)
            
            if not active_vectors:
                active_vectors.append(v)
                continue
                
            # Check against active_vectors
            # To speed up, we can check in batches?
            # Or just use numpy if active_vectors is an array.
            # Maintaining active_vectors as a list and converting to array every time is O(M^2) memory.
            # Maintaining as array and vstack is O(M^2) memory copy.
            
            # Let's try to keep active_vectors as a numpy array.
            # But we are in a loop.
            
            # Alternative: Keep active_vectors as a list, and only convert to array if len > threshold?
            # No, dot product with list is slow.
            
            # Let's stick to the array approach but be careful.
            # Actually, we can accumulate in a list and flush to array periodically?
            # No, we need to check against ALL previous.
            
            # Let's use a numpy array for active_vectors.
            # To avoid vstack overhead, we can pre-allocate?
            # Max size is ~40000.
            # Pre-allocate array of size 50000 x 11.
            # Keep track of count.
            
            # This is a significant optimization.
            
            pass # placeholder for logic
            
        # End of run
        current_count = len(type1_vectors) + len(active_vectors)
        if current_count > best_count:
            best_count = current_count
            best_vectors = type1_vectors + active_vectors

    return np.array(best_vectors, dtype=np.int64)
