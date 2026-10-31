import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin

    # 1. Generate Type A vectors (norm squared 4, one non-zero entry is +/-2)
    type_a_vectors = []
    for i in range(D):
        for s in (2, -2):
            v = [0] * D
            v[i] = s
            type_a_vectors.append(tuple(v))
    
    # 2. Generate Type B vectors (norm squared 4, four non-zero entries are +/-1)
    type_b_vectors = []
    # Pre-compute all combinations of 4 indices
    indices_combos = list(itertools.combinations(range(D), 4))
    for idxs in indices_combos:
        # 2^4 = 16 sign combinations
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            type_b_vectors.append(tuple(v))
            
    # 3. Greedy selection of Type B vectors
    # We want to avoid sharing any (3 indices, 3 signs) pattern.
    # Key: (sorted_indices, sign_tuple)
    
    def get_keys(v):
        # v is a tuple of length D
        non_zero_indices = [i for i in range(D) if v[i] != 0]
        keys = set()
        # Choose 3 indices from the 4 non-zero ones
        for combo in itertools.combinations(range(4), 3):
            idxs = [non_zero_indices[j] for j in combo]
            signs = [v[i] for i in idxs]
            # Sort indices for consistent key
            sorted_pairs = sorted(zip(idxs, signs))
            key = (tuple(p[0] for p in sorted_pairs), tuple(p[1] for p in sorted_pairs))
            keys.add(key)
        return keys

    # Pre-compute keys for all Type B vectors
    type_b_with_keys = []
    for v in type_b_vectors:
        keys = get_keys(v)
        type_b_with_keys.append((v, keys))

    best_type_b = []
    best_len = 0
    
    # Try a few random orders to find a large independent set
    # 5280 items is small, so we can try many permutations
    num_trials = 20
    for _ in range(num_trials):
        if time.time() > deadline:
            break
            
        random.shuffle(type_b_with_keys)
        used_keys = set()
        current_set = []
        
        for v, keys in type_b_with_keys:
            # Check if any key is already used
            if not any(k in used_keys for k in keys):
                current_set.append(v)
                used_keys.update(keys)
                
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_type_b = current_set
            
    # Combine Type A and best Type B
    all_vectors = type_a_vectors + best_type_b
    
    # Convert to numpy array
    if not all_vectors:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    return np.array(all_vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
