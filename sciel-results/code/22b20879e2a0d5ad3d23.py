"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # Strategy:
    # 1. Include all 22 Type A vectors (weight 1, values +/-2).
    #    They are mutually compatible and compatible with all Type C.
    # 2. Find a large set of mutually compatible Type C vectors (weight 4, values +/-1).
    #    Constraint: dot product <= 2.
    
    # Generate Type A vectors
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    # Generate Type C vectors
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1

    rng = np.random.default_rng(42)
    
    best_size = 0
    best_v4_indices = []
    
    # Greedy Randomized Construction with Restarts
    # We shuffle the candidates and greedily pick compatible ones.
    # Compatibility check: max dot product with current set <= 2.
    
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Shuffle indices
        order = rng.permutation(N4)
        
        selected_indices = []
        selected_vectors = [] # List of np arrays or just keep indices and use vectors_v4
        
        for idx in order:
            v = vectors_v4[idx]
            
            # Check compatibility with all selected
            # If no selected, it's compatible
            if not selected_vectors:
                selected_vectors.append(v)
                selected_indices.append(idx)
                continue
            
            # Vectorized check: dot product of v with all selected
            # selected_vectors is a list, convert to array for speed if large?
            # Or just loop if small. 
            # To be fast, we can maintain a numpy array of selected vectors.
            
            # Efficient check:
            # Create matrix of selected vectors (K x 11)
            # dot = selected_matrix @ v
            # if max(dot) <= 2: add
            
            # To avoid creating a new matrix every time, we can just check against the last few?
            # No, must check all.
            
            # Optimization: Keep selected_vectors as a list, but if it grows, use numpy.
            # Let's just use numpy array for selected vectors.
            
            if len(selected_vectors) == 0:
                 selected_vectors = [v]
                 selected_indices.append(idx)
            else:
                 # Convert list to array for dot product? 
                 # It's better to maintain a numpy array `sel_arr`
                 pass
        
        # Re-implement inner loop with numpy array for speed
        selected_arr = np.empty((0, D), dtype=np.int64)
        selected_indices = []
        
        for idx in order:
            v = vectors_v4[idx]
            if len(selected_arr) == 0:
                selected_arr = v.reshape(1, -1)
                selected_indices.append(idx)
            else:
                dots = selected_arr @ v
                if np.max(dots) <= 2:
                    selected_arr = np.vstack([selected_arr, v])
                    selected_indices.append(idx)
        
        current_size = len(selected_indices)
        if current_size > best_size:
            best_size = current_size
            best_v4_indices = selected_indices.copy()
            
    # Construct final result
    result_v4 = vectors_v4[best_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
