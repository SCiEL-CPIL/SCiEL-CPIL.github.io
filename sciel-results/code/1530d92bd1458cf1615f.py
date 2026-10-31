import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_vectors = None
    best_count = 0
    
    # We will try different weights
    # Weight 2 is the seed, we know it's 220.
    # We focus on Weight 3 and 4 which might yield more.
    
    for weight in [3, 4]:
        # Generate all vectors of given weight
        # Indices: combinations of D, choose weight
        # Signs: 2^weight combinations of 1, -1
        
        vectors = []
        for idxs in itertools.combinations(range(D), weight):
            for signs in itertools.product([1, -1], repeat=weight):
                v = np.zeros(D, dtype=np.int64)
                for idx, s in zip(idxs, signs):
                    v[idx] = s
                vectors.append(v)
        
        A = np.array(vectors)
        N = len(A)
        
        # Compute dot product matrix
        # M[i, j] = A[i] . A[j]
        M = A @ A.T
        
        # Threshold
        # Condition: u . v <= weight / 2
        # If weight is even, weight/2 is integer.
        # If weight is odd, weight/2 is x.5. Since u.v is integer, u.v <= floor(weight/2).
        threshold = weight // 2
        
        # We want to select max subset of rows such that M[i, j] <= threshold for all i != j
        # This is Max Independent Set on graph where edge (i, j) exists if M[i, j] > threshold
        
        # Greedy with random restarts
        local_best = 0
        local_best_idx = None
        
        while time.time() < deadline:
            order = np.random.permutation(N)
            selected = []
            selected_set = set()
            
            for i in order:
                # Check compatibility with all selected
                # Optimization: keep a boolean mask of compatible indices?
                # Or just check against selected.
                # Since N is small (~5000), and selected is likely large, 
                # checking against selected is O(K). Total O(N*K).
                # K can be up to N. O(N^2) per restart.
                # 5000^2 = 25M ops. In Python, this might be slow.
                # 1 restart might take 1-2 seconds?
                # We can do maybe 50-100 restarts.
                
                # Faster check:
                # Pre-compute conflicts for i?
                # conflicts[i] = indices j where M[i, j] > threshold
                # Then check if any j in selected is in conflicts[i].
                
                # Let's precompute conflict sets for speed
                # This is heavy memory if we store lists for all N.
                # But N=5000, M is 5000x5000.
                # We can just check on the fly using numpy slicing if we maintain a mask?
                
                # Let's try a simpler greedy:
                # Maintain a mask of "available" nodes.
                # Initially all True.
                # Pick a node, remove it and its neighbors from available.
                
                available = np.ones(N, dtype=bool)
                count = 0
                selected_indices = []
                
                for i in order:
                    if not available[i]:
                        continue
                    
                    # Select i
                    selected_indices.append(i)
                    count += 1
                    available[i] = False
                    
                    # Find neighbors of i that are currently available
                    # M[i, :] > threshold
                    # We only care about those in available
                    # neighbors = np.where((M[i, :] > threshold) & available)[0]
                    # available[neighbors] = False
                    
                    # Vectorized update
                    conflict_mask = M[i, :] > threshold
                    # Only update those that are available
                    # We can just do:
                    available[conflict_mask & available] = False
                    # Note: available[i] is already False.
                    
                if count > local_best:
                    local_best = count
                    local_best_idx = selected_indices
                    # Optional: break if we reached theoretical limit? No limit known.
                
                # If time is running out, break restart loop
                if time.time() > deadline:
                    break
            
        if local_best > best_count:
            best_count = local_best
            best_vectors = A[local_best_idx]
            
    # Fallback to seed if nothing found (should not happen)
    if best_vectors is None:
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        best_vectors = np.array(vectors, dtype=np.int64)
        
    return best_vectors
