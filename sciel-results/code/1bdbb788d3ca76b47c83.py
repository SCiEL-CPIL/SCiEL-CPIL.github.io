import numpy as np
import itertools
import time

D = 11
DEADLINE = 110.0 # Safety margin

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    
    vectors = []
    
    # Part 1: All 220 vectors of Type 1 (Two +/- 2s)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                vectors.append(v)
                
    # Part 2: Greedy selection of Type 2 vectors (Eight +/- 1s)
    # Generate all Type 2 vectors
    all_type2 = []
    for support in itertools.combinations(range(D), 8):
        base = [0] * D
        for idx in support:
            base[idx] = 1
        for signs in itertools.product([-1, 1], repeat=8):
            v = base.copy()
            for k, idx in enumerate(support):
                v[idx] = signs[k]
            all_type2.append(v)
            
    V2 = np.array(all_type2, dtype=np.int8)
    
    # Shuffle with a fixed seed for reproducibility
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(V2))
    V2 = V2[perm]
    
    accepted_indices = []
    V_accepted = np.empty((0, D), dtype=np.int8)
    
    chunk_size = 2048
    num_chunks = (len(V2) + chunk_size - 1) // chunk_size
    
    for c in range(num_chunks):
        if time.time() > DEADLINE:
            break
            
        start_idx = c * chunk_size
        end_idx = min(start_idx + chunk_size, len(V2))
        V_chunk = V2[start_idx:end_idx]
        
        # Check compatibility with already accepted vectors
        if len(V_accepted) > 0:
            # dots shape: (chunk_size, num_accepted)
            dots = V_chunk @ V_accepted.T
            valid_mask = np.all(dots <= 4, axis=1)
        else:
            valid_mask = np.ones(len(V_chunk), dtype=bool)
        
        # Get indices of valid candidates in the chunk
        valid_local_indices = np.where(valid_mask)[0]
        
        # Now, among these valid candidates, select a subset that is mutually compatible
        # We do this greedily. Since the chunk is small and conflicts are rare,
        # we can check against the vectors added in this specific chunk step-by-step.
        
        V_new_in_chunk = []
        
        for local_idx in valid_local_indices:
            if time.time() > DEADLINE:
                break
            
            v = V_chunk[local_idx]
            
            # Check against V_new_in_chunk
            # V_new_in_chunk is a list of np arrays
            compatible = True
            for v_new in V_new_in_chunk:
                dot = np.dot(v, v_new)
                if dot > 4:
                    compatible = False
                    break
            
            if compatible:
                V_new_in_chunk.append(v)
                accepted_indices.append(start_idx + local_idx)
                
        # Update V_accepted
        if V_new_in_chunk:
            V_new_arr = np.array(V_new_in_chunk, dtype=np.int8)
            V_accepted = np.vstack([V_accepted, V_new_arr])
            
    # Combine Type 1 and Type 2 vectors
    if accepted_indices:
        V_type2_selected = V2[accepted_indices]
        vectors.extend(V_type2_selected.tolist())
        
    return np.array(vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
