"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 100.0  # 100s to be safe within 120s limit
    
    # 1. Generate Type B vectors: norm^2 = 4, one entry is +/-2
    # These are compatible with all Type A vectors.
    vecs_B = []
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            vecs_B.append(v)
            
    vecs_B = np.array(vecs_B, dtype=np.int64)
    
    # 2. Generate Type A vectors: norm^2 = 4, four entries are +/-1
    vecs_A_list = []
    indices = list(range(D))
    for comb in itertools.combinations(indices, 4):
        for signs in itertools.product((1, -1), repeat=4):
            v = [0] * D
            for idx, s in zip(comb, signs):
                v[idx] = s
            vecs_A_list.append(v)
            
    vecs_A = np.array(vecs_A_list, dtype=np.int64)
    
    # Shuffle to avoid ordering bias
    np.random.shuffle(vecs_A)
    
    selected_A = []
    selected_A_arr = None
    
    for v in vecs_A:
        if time.time() > deadline:
            break
            
        if not selected_A:
            selected_A.append(v)
            selected_A_arr = np.array([v])
            continue
            
        # Check compatibility: max dot product must be <= 2
        # If selected_A_arr is large, this is O(N)
        dots = selected_A_arr @ v
        if np.max(dots) <= 2:
            selected_A.append(v)
            # Efficiently update array
            if selected_A_arr.shape[0] == 1:
                selected_A_arr = np.vstack([selected_A_arr, v])
            else:
                selected_A_arr = np.concatenate([selected_A_arr, v.reshape(1, -1)])
                
    # Combine results
    if selected_A:
        final_vecs = np.vstack([np.array(selected_A, dtype=np.int64), vecs_B])
    else:
        final_vecs = vecs_B
        
    return final_vecs
# EVOLVE-BLOCK-END
