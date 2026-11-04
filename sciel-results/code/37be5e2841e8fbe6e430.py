import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Generate all 4-subsets of {0, ..., 10}
    all_supports = list(itertools.combinations(range(D), 4))
    n_supports = len(all_supports)  # 330
    
    # Convert to bitmasks for fast intersection checks
    support_masks = np.array([sum(1 << i for i in s) for s in all_supports], dtype=np.int64)
    
    # Precompute popcount lookup for 11-bit numbers
    popcount_lut = np.array([bin(i).count('1') for i in range(2048)], dtype=np.int8)
    
    # Find a large set of supports with pairwise intersection <= 2
    # Use greedy with random restarts
    best_support_set = []
    best_support_size = 0
    
    rng = np.random.default_rng(42)
    num_restarts = 200
    
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        # Randomly shuffle supports
        shuffled = list(all_supports)
        rng.shuffle(shuffled)
        
        current_set = []
        current_masks = []
        
        for sup in shuffled:
            mask = sum(1 << i for i in sup)
            valid = True
            for cm in current_masks:
                inter = popcount_lut[mask & cm]
                if inter > 2:
                    valid = False
                    break
            if valid:
                current_set.append(sup)
                current_masks.append(mask)
        
        if len(current_set) > best_support_size:
            best_support_size = len(current_set)
            best_support_set = current_set[:]
    
    # Now for each selected support, generate all 16 sign vectors
    # and greedily select those that satisfy the constraint with all previously selected vectors
    selected_vectors = []
    selected_arr = np.array([]).reshape(0, D)
    
    for sup in best_support_set:
        # Generate all 16 sign vectors for this support
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(sup, signs):
                v[idx] = sign
            v_arr = np.array(v, dtype=np.int64)
            
            if len(selected_arr) == 0:
                selected_vectors.append(v)
                selected_arr = v_arr.reshape(1, D)
            else:
                # Check dot products with all selected vectors
                # Constraint: u·v <= 2 for all pairs
                dots = selected_arr @ v_arr
                if np.all(dots <= 2):
                    selected_vectors.append(v)
                    selected_arr = np.vstack([selected_arr, v_arr.reshape(1, D)])
        
        if time.time() > deadline - 5:
            break
    
    return np.array(selected_vectors, dtype=np.int64)
