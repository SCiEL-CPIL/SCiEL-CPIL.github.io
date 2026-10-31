"""Constructing a kissing configuration in 11 dimensions using optimized bitmask greedy selection."""

import itertools
import numpy as np

D = 11

def construct():
    # Type A vectors: Norm squared 4, one entry is +/-2.
    # There are 2 * D = 22 such vectors.
    # They are always valid against each other (dot product is 0 or -4 or 4? 
    # Wait. v_i = (0,..,2,..0), v_j = (0,..,-2,..0). Dot = -4. Distance squared = 4+4 - 2(-4) = 16 >= 4. OK.
    # v_i and -v_i: Dot = -4. Distance squared = 16. OK.
    # So all Type A vectors are mutually compatible.
    
    vectors_a = []
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * s
            vectors_a.append(v)
            
    # Type B vectors: Norm squared 4, four entries are +/-1.
    # Total candidates: C(11,4) * 2^4 = 330 * 16 = 5280.
    
    # Pre-compute basis indices and signs for Type B
    # We will represent these as (mask, sign_bits) or just full vectors for the final output,
    # but for selection, we can use a fast dot product method.
    
    # Let's generate all Type B vectors as an array for batch processing.
    # Shape: (N_cand, D)
    
    indices_list = list(itertools.combinations(range(D), 4))
    signs_list = list(itertools.product([1, -1], repeat=4))
    
    N_cand = len(indices_list) * len(signs_list)
    candidates = np.zeros((N_cand, D), dtype=np.int8) # int8 is enough for values -1, 0, 1
    
    idx = 0
    for comb in indices_list:
        for signs in signs_list:
            for k, pos in enumerate(comb):
                candidates[idx, pos] = signs[k]
            idx += 1
            
    # Greedy Selection Process
    # We maintain `selected` as a list of indices into `candidates` that have been accepted.
    # We process candidates in chunks.
    
    selected_indices = []
    selected_vecs = [] # List of np arrays for easy appending to final result
    
    # To speed up, we can sort candidates. 
    # A good heuristic for kissing numbers is to process vectors in a way that spreads them out.
    # However, simple sequential is often decent if the batch size is large.
    
    # Optimization: We will check compatibility against `selected_vecs`.
    # We'll keep `selected_vecs` as a NumPy array (M, D) for vectorized dot products.
    # We update this array after each batch.
    
    batch_size = 512
    num_batches = (N_cand + batch_size - 1) // batch_size
    
    # We need to be careful: if we accept multiple vectors from a batch, they must be mutually compatible.
    # Strategy:
    # 1. Compute dots of entire batch against current selected.
    # 2. Filter to `valid_batch` where all dots <= 2.
    # 3. Within `valid_batch`, perform a greedy sub-selection (O(N^2) dot products within the batch, which is small).
    # 4. Add these to `selected`.
    
    current_selected = np.empty((0, D), dtype=np.int8)
    
    for b in range(num_batches):
        start = b * batch_size
        end = min(start + batch_size, N_cand)
        batch = candidates[start:end]
        
        if current_selected.shape[0] == 0:
            # If nothing selected yet, we can just pick the first one? 
            # No, we should still do local greedy to maximize.
            # But if empty, all are valid against selected.
            pass
            
        if current_selected.shape[0] > 0:
            # Compute dot products: (B, D) @ (D, M) -> (B, M)
            dots = batch @ current_selected.T
            # Check if all dots <= 2
            # valid_mask[i] is True if candidate i is compatible with ALL selected
            valid_mask = np.all(dots <= 2, axis=1)
        else:
            valid_mask = np.ones(len(batch), dtype=bool)
            
        valid_indices_in_batch = np.where(valid_mask)[0]
        
        if len(valid_indices_in_batch) == 0:
            continue
            
        # Get the vectors
        valid_vecs = batch[valid_indices_in_batch]
        
        # Now we need to select a maximal subset from `valid_vecs` that are mutually compatible.
        # This is a maximum independent set problem on a graph, which is NP-hard,
        # but for a batch of ~500, a simple greedy works well.
        
        local_selected = []
        # Iterate through valid_vecs in order
        for i in range(len(valid_vecs)):
            v = valid_vecs[i]
            # Check compatibility with previously accepted in this local batch
            if not local_selected:
                local_selected.append(v)
                continue
                
            local_sel_arr = np.array(local_selected)
            dots_local = local_sel_arr @ v
            if np.all(dots_local <= 2):
                local_selected.append(v)
                
        # Add local_selected to global selected
        if local_selected:
            local_selected_arr = np.array(local_selected, dtype=np.int8)
            if current_selected.shape[0] == 0:
                current_selected = local_selected_arr
            else:
                current_selected = np.vstack([current_selected, local_selected_arr])
            selected_vecs.extend(local_selected)
            
    # Combine Type A and Type B
    # We must verify that Type B vectors are compatible with Type A vectors.
    # The greedy above only checked B against B.
    # We need to filter the final `selected_vecs` against `vectors_a` as well.
    
    final_vectors = []
    
    # Check Type A
    a_arr = np.array(vectors_a, dtype=np.int8)
    
    # Re-filter the selected B vectors against A vectors
    # Since `selected_vecs` is a list of arrays, let's convert to array
    if selected_vecs:
        b_arr = np.array(selected_vecs, dtype=np.int8)
        # Dots (N_b, D) @ (D, N_a) -> (N_b, N_a)
        dots_ba = b_arr @ a_arr.T
        # A vector b is valid if min(dot(b, a)) <= 2 for all a?
        # Condition: dist^2 >= max(norm^2). 
        # For integer vectors with norm^2=4, dist^2 = 8 - 2*dot.
        # We need dist^2 >= 4 => 8 - 2*dot >= 4 => 2*dot <= 4 => dot <= 2.
        # So we need dot(b, a) <= 2 for ALL a in A.
        valid_b_mask = np.all(dots_ba <= 2, axis=1)
        
        for i, v in enumerate(b_arr):
            if valid_b_mask[i]:
                final_vectors.append(v)
    else:
        valid_b_mask = np.array([])
        
    # Add all Type A vectors
    for v in a_arr:
        final_vectors.append(v)
        
    if not final_vectors:
        return np.array([])
        
    result = np.array(final_vectors, dtype=np.int64)
    
    return result
