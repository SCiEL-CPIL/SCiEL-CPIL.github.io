import itertools
import numpy as np
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    best_vectors = None
    best_score = 0
    
    # Weights to try
    # Weight 2 is known to be 220
    # We will try 2, 3, 4, 5
    
    for weight in [2, 3, 4, 5]:
        if time.time() > deadline:
            break
            
        # Generate vectors
        vecs = []
        for indices in itertools.combinations(range(D), weight):
            for signs in itertools.product([1, -1], repeat=weight):
                v = np.zeros(D, dtype=np.int64)
                for idx, sign in zip(indices, signs):
                    v[idx] = sign
                vecs.append(v)
        
        if not vecs:
            continue
            
        vecs = np.array(vecs, dtype=np.int64)
        n = len(vecs)
        
        # Calculate threshold
        # d^2 >= W  =>  2W - 2 dot >= W  =>  2 dot <= W  => dot <= W/2
        # Since dot is integer, dot <= floor(W/2)
        max_dot = weight // 2
        
        # Greedy Independent Set
        # To speed up, we can use a random order
        # And maintain a set of "available" vectors
        
        # For small N (< 5000), we can use a simple list
        # For larger N, we might need optimization, but 14784 is okay if sparse
        
        # Optimization: Pre-compute dot products? No, memory.
        # Use random greedy:
        # 1. Shuffle indices
        # 2. Iterate. If vector is compatible with all selected, add it.
        
        # To make it faster, we can try multiple random shuffles
        num_trials = 50 if n < 2000 else 10
        
        best_local = 0
        best_local_vecs = None
        
        for _ in range(num_trials):
            if time.time() > deadline:
                break
                
            indices = np.random.permutation(n)
            selected = []
            # We check against all selected.
            # To speed up, we can break early if conflict found.
            
            for idx in indices:
                v = vecs[idx]
                is_ok = True
                for s in selected:
                    # Dot product
                    # If we store selected as array, we can vectorize?
                    # But selected grows.
                    # Let's just loop.
                    if np.dot(v, s) > max_dot:
                        is_ok = False
                        break
                if is_ok:
                    selected.append(v)
            
            if len(selected) > best_local:
                best_local = len(selected)
                best_local_vecs = np.array(selected, dtype=np.int64)
        
        if best_local > best_score:
            best_score = best_local
            best_vectors = best_local_vecs
    
    if best_vectors is None:
        # Fallback to seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        best_vectors = np.array(vectors, dtype=np.int64)
        
    return best_vectors
# EVOLVE-BLOCK-END
