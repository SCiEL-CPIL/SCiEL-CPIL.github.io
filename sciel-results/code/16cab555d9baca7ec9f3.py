import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s limit

    # 1. Generate Type A vectors (one non-zero entry +/- 2)
    type_a_vectors = []
    for i in range(D):
        for sign in (2, -2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = sign
            type_a_vectors.append(v)
    type_a_vectors = np.array(type_a_vectors)

    # 2. Generate Type B vectors (four non-zero entries +/- 1)
    type_b_vectors = []
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, sign in zip(support, signs):
                v[idx] = sign
            type_b_vectors.append(v)
    type_b_vectors = np.array(type_b_vectors)

    n_b = len(type_b_vectors)

    # 3. Compute incompatibility graph for Type B
    dot_products = type_b_vectors @ type_b_vectors.T
    adjacency = (dot_products >= 3)
    np.fill_diagonal(adjacency, False)

    # Precompute degree for each vertex using NumPy for speed and correct types
    degrees = np.sum(adjacency, axis=1).astype(np.int64)

    # Try multiple strategies to find a good MIS
    best_mis_indices = []
    best_mis_size = 0

    # Strategy 1: Min-degree greedy
    # This is O(N^2) but vectorized updates make it fast for N~5000
    available = np.ones(n_b, dtype=bool)
    selected_indices = []
    while np.any(available):
        if time.time() > deadline:
            break
        avail_indices = np.where(available)[0]
        if len(avail_indices) == 0:
            break
        
        # Get degrees of available vertices
        avail_degrees = degrees[avail_indices]
        min_deg_idx_in_avail = np.argmin(avail_degrees)
        chosen_global_idx = avail_indices[min_deg_idx_in_avail]
        
        selected_indices.append(int(chosen_global_idx))
        
        # Update available mask
        available[chosen_global_idx] = False
        neighbors = np.where(adjacency[chosen_global_idx])[0]
        available[neighbors] = False

    if len(selected_indices) > best_mis_size:
        best_mis_size = len(selected_indices)
        best_mis_indices = selected_indices[:]

    # Strategy 2: Vectorized Random Greedy
    # Multiple restarts with random ordering, using vectorized compatibility check
    rng = np.random.default_rng(42)
    num_trials = 50 # More trials, but each is O(N) vectorized
    
    for trial in range(num_trials):
        if time.time() > deadline - 5:
            break
        
        order = rng.permutation(n_b)
        selected = []
        # compatible_with_selected starts as all True
        compatible_with_selected = np.ones(n_b, dtype=bool)
        
        for idx in order:
            if time.time() > deadline - 5:
                break
            
            # If idx is compatible with all currently selected, add it
            if compatible_with_selected[idx]:
                selected.append(int(idx))
                # Update compatibility: remove idx and its neighbors from future candidates
                compatible_with_selected[idx] = False
                compatible_with_selected[adjacency[idx]] = False
                
        if len(selected) > best_mis_size:
            best_mis_size = len(selected)
            best_mis_indices = selected[:]

    # Convert selected indices to vectors
    selected_b_vectors = type_b_vectors[best_mis_indices]

    # Combine with Type A
    all_vectors = np.vstack([type_a_vectors, selected_b_vectors])

    return all_vectors
