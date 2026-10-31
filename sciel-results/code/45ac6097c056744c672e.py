import numpy as np
import time
import random
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    # Generate all 4-sparse +/-1 vectors
    # Indices 0..10
    # Choose 4 indices, then 4 signs
    candidates_4 = []
    for indices in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(indices, signs):
                v[idx] = s
            candidates_4.append(v)
            
    # Convert to numpy for potential vectorization, but list of lists is fine for greedy
    # Actually, let's keep as list of tuples for hashing? No, we need to check distance.
    
    # Also generate the 22 vectors of form +/- 2 e_i
    candidates_2 = []
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates_2.append(v)
            
    best_set = []
    
    # We will run greedy multiple times
    # To speed up, we can shuffle candidates_4
    
    for seed in range(100):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        random.shuffle(candidates_4)
        
        current_set = []
        # We will add the 22 vectors at the end if they are compatible.
        # But they are compatible with ALL 4-sparse vectors.
        # So we just need to find a large set of 4-sparse vectors.
        
        for v in candidates_4:
            if time.time() > deadline:
                break
                
            compatible = True
            for u in current_set:
                # Check dot product <= 2
                # v and u are sparse
                # Dot product
                dot = 0
                # Since they are sparse, we can iterate over non-zeros
                # But D=11 is small, just sum
                # Optimization: precompute non-zero indices?
                # For now, simple loop
                for k in range(D):
                    dot += v[k] * u[k]
                if dot > 2:
                    compatible = False
                    break
                    
            if compatible:
                current_set.append(v)
                
        # Add the 22 vectors
        final_set = current_set + candidates_2
        
        if len(final_set) > len(best_set):
            best_set = final_set
            if len(best_set) > 300: # Early stop if we are way above target
                 break

    # Convert to numpy array
    if not best_set:
        # Fallback to seed program result
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    return np.array(best_set, dtype=np.int64)
