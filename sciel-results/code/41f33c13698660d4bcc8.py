import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin

    # 1. Generate Type A vectors (one non-zero entry +/- 2)
    # There are 22 such vectors.
    # They are compatible with all Type B vectors and with each other.
    type_a_vectors = []
    for i in range(D):
        for sign in (2, -2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = sign
            type_a_vectors.append(v)
    type_a_vectors = np.array(type_a_vectors)
    
    # 2. Generate Type B vectors (four non-zero entries +/- 1)
    # C(11, 4) * 2^4 = 330 * 16 = 5280 vectors.
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
    # Incompatible if dot product >= 3.
    # Dot product matrix
    dot_products = type_b_vectors @ type_b_vectors.T
    
    # Create adjacency mask
    # We need to exclude self (dot=4) but self is not a neighbor in MIS context?
    # In MIS, we pick a vertex, remove it and its neighbors.
    # Self is not a neighbor.
    # Incompatible if dot >= 3.
    # Note: dot=4 only for self.
    # So for i != j, dot >= 3 means dot == 3.
    
    adjacency = (dot_products >= 3)
    # Zero out diagonal
    np.fill_diagonal(adjacency, False)
    
    # Initial degrees
    degrees = adjacency.sum(axis=1)
    
    # 4. Greedy MIS
    available = np.ones(n_b, dtype=bool)
    selected_indices = []
    
    # To speed up, we can pre-compute the neighbor lists as lists of indices
    # But boolean mask is fine.
    
    # Optimization: 
    # Instead of scanning all 5280 every time, we can maintain a list of candidates.
    # But 5280 is small enough that a full scan might be fast enough in NumPy.
    # Let's try full scan first.
    
    while np.any(available):
        if time.time() > deadline:
            break
            
        # Find available vertices
        avail_indices = np.where(available)[0]
        if len(avail_indices) == 0:
            break
            
        # Get degrees of available vertices
        avail_degrees = degrees[avail_indices]
        
        # Pick the one with minimum degree
        min_deg_idx_in_avail = np.argmin(avail_degrees)
        chosen_global_idx = avail_indices[min_deg_idx_in_avail]
        
        selected_indices.append(chosen_global_idx)
        
        # Remove chosen and its neighbors
        available[chosen_global_idx] = False
        neighbors = np.where(adjacency[chosen_global_idx])[0]
        available[neighbors] = False
        
        # Note: We are NOT updating degrees dynamically.
        # This is "Static Min-Degree".
        # If we wanted dynamic, we'd need to update degrees of remaining vertices.
        # Let's stick to static first. If score is low, we can try dynamic or random restarts.
        
    # Convert selected indices to vectors
    selected_b_vectors = type_b_vectors[selected_indices]
    
    # Combine with Type A
    all_vectors = np.vstack([type_a_vectors, selected_b_vectors])
    
    return all_vectors
