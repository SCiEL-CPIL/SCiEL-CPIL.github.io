import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    D = 11
    
    # Type A: Vectors with one non-zero entry +/- 2
    # Norm squared = 4
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)
        
    # Type D: Vectors with four non-zero entries +/- 1
    # Norm squared = 4
    vectors_D_list = []
    indices = list(range(D))
    for combo in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(combo, signs):
                v[idx] = sign
            vectors_D_list.append(v)
            
    vectors_D = np.array(vectors_D_list, dtype=np.int8)
    N_D = len(vectors_D)
    
    # Compute conflict matrix for Type D
    # Conflict if dot product >= 3 (distance squared = 8 - 2*dot <= 2 < 4)
    dot_products = vectors_D @ vectors_D.T
    conflict_mask = dot_products >= 3
    
    # Precompute neighbor lists for speed
    neighbor_lists = []
    for i in range(N_D):
        conflicts = np.where(conflict_mask[i])[0]
        neighbor_lists.append(conflicts)
        
    best_set_D = []
    best_size = 0
    
    # Precompute degree for each node (number of conflicts)
    degrees = np.array([len(nl) for nl in neighbor_lists])
    
    # Greedy Independent Set with Random Restarts and minimum-degree heuristic
    # We use a hybrid approach: sample a small subset of available nodes and pick the one with min degree
    # This is faster than scanning all available nodes each step, but still guided by structure
    
    # We'll run as many restarts as possible within the time budget
    # Each restart is O(N * avg_degree) roughly, but with sampling it's O(N * sample_size)
    
    # Estimate time per restart: N_D is C(11,4)*16 = 330*16 = 5280
    # Actually C(11,4) = 330, times 16 sign combos = 5280 vectors
    # That's a lot. Let's check: 11 choose 4 is 330. 330 * 16 = 5280.
    
    # The conflict matrix is 5280x5280, which is ~28M entries. That might be heavy.
    # Let's be careful with memory and time.
    
    # Actually, let's just run the loop and break when time is up.
    
    num_restarts = 500  # Will break early if time runs out
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        available = set(range(N_D))
        current_set = []
        
        while available:
            if len(available) == 1:
                v_idx = next(iter(available))
            else:
                # Sample a small subset to find a good candidate (min degree)
                sample_size = min(30, len(available))
                # Convert to list for sampling
                avail_list = list(np.random.choice(list(available), size=sample_size, replace=False))
                degs = degrees[avail_list]
                min_local = np.argmin(degs)
                v_idx = avail_list[min_local]
            
            current_set.append(v_idx)
            available.remove(v_idx)
            
            # Remove all neighbors from available
            available.difference_update(neighbor_lists[v_idx])
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set_D = current_set
    
    # Final polish: try to add more Type D vectors to the best set
    best_set_set = set(best_set_D)
    for v_idx in range(N_D):
        if v_idx in best_set_set:
            continue
        # Check if v_idx can be added
        can_add = True
        for c in neighbor_lists[v_idx]:
            if c in best_set_set:
                can_add = False
                break
        if can_add:
            best_set_D.append(v_idx)
            best_set_set.add(v_idx)
    
    # Combine Type A and selected Type D
    vectors_A = np.array(vectors_A, dtype=np.int64)
    vectors_D_selected = vectors_D[best_set_D].astype(np.int64)
    
    final_vectors = np.vstack([vectors_A, vectors_D_selected])
    
    return final_vectors
