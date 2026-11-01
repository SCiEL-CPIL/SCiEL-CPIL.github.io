import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Generate all 4-subsets of range(11)
    # Represent as tuple of 4 ints
    all_subsets = list(itertools.combinations(range(D), 4))
    n_subsets = len(all_subsets)
    
    # Precompute intersection sizes? 
    # Or just compute on fly. 330*330 is small.
    
    best_set = []
    
    # Greedy with random restarts
    import random
    
    max_restarts = 1000 # Will be limited by time
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Shuffle
        candidates = all_subsets[:]
        random.shuffle(candidates)
        
        current_set = []
        
        for c in candidates:
            # Check compatibility
            compatible = True
            for s in current_set:
                # Count intersection
                # c and s are tuples of 4 ints
                inter = len(set(c) & set(s))
                if inter > 2:
                    compatible = False
                    break
            if compatible:
                current_set.append(c)
                
        if len(current_set) > len(best_set):
            best_set = current_set
            # Early stop if we found a very good one? 
            # Max possible is bounded. 
            # If we hit 30+, it's good.
            
    # Construct vectors
    vectors = []
    for subset in best_set:
        # Generate all 16 sign combinations
        # Subset is a tuple of 4 indices
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(subset, signs):
                v[idx] = sign
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
