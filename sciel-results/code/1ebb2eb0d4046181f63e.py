import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    D = 11

    # Type A: Vectors with one non-zero entry +/- 2
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)
    vectors_A = np.array(vectors_A, dtype=np.int64)

    # Type D: Vectors with four non-zero entries +/- 1
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    
    # Generate all Type D vectors
    vectors_D_list = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(s, signs):
                v[idx] = sign
            vectors_D_list.append(v)
    
    vectors_D = np.array(vectors_D_list, dtype=np.int64)
    N_D = len(vectors_D)
    
    # Compute conflict matrix for Type D vectors
    # Conflict if dot product >= 3
    # Vectors are in {-1, 0, 1}
    dots = vectors_D @ vectors_D.T
    conflict = dots >= 3
    
    # Find a large independent set in the conflict graph
    # Use randomized greedy multiple times
    rng = np.random.RandomState(42)
    py_rng = random.Random(42)
    
    best_selected = []
    best_size = 0
    
    # Number of greedy attempts
    # Each attempt is O(N * degree). N=5280.
    # Should be very fast.
    num_attempts = 1000
    
    for i in range(num_attempts):
        if time.time() > deadline - 10:
            break
            
        # Randomize the order of processing
        order = np.arange(N_D)
        rng.shuffle(order)
        
        available = np.ones(N_D, dtype=bool)
        selected = []
        
        for v in order:
            if not available[v]:
                continue
            selected.append(v)
            available[v] = False
            # Mark neighbors as unavailable
            # conflict[v] is a boolean array
            neighbors = np.where(conflict[v])[0]
            available[neighbors] = False
            
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected
            
        # Early stop if we find a very large set?
        # The max independent set size is unknown, but likely a few hundred.
        # If we find > 500, we can stop?
        # Let's just run all attempts or until time limit.
        
    # Construct final vectors
    final_vectors_D = vectors_D[best_selected]
    
    # Combine A and D
    final_vectors = np.vstack([vectors_A, final_vectors_D])
    
    return final_vectors
