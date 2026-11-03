import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Generate all weight-3 vectors
    # Support is a tuple of 3 indices
    # Signs are 3 bits
    vectors = []
    indices_map = {}
    
    for support in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            vectors.append(v)
            
    n_vecs = len(vectors)
    
    # Compute conflict matrix
    # Conflict if dot product > 1 (since R^2=3, limit is 1.5, integer dot <= 1)
    # Dot product of two weight-3 vectors:
    # If supports are disjoint: 0
    # If share 1: +/- 1
    # If share 2: +/- 2 or 0
    # If share 3: +/- 3, +/- 1
    # Conflict if dot >= 2.
    
    # We can compute dot products efficiently
    V = np.array(vectors)
    
    # Calculate all pairwise dot products
    # This is a 1320x1320 matrix. 1320^2 * 8 bytes ~ 14 MB. Fast.
    dots = V @ V.T
    
    # Conflict mask: dot > 1
    conflict_mask = dots > 1
    np.fill_diagonal(conflict_mask, False) # Self is not a conflict for independent set logic (we pick distinct)
    
    # Greedy Independent Set
    best_set_size = 0
    best_indices = []
    
    np.random.seed(42)
    
    # We can optimize by sorting nodes by degree?
    # But random restarts are better for finding large sets in structured graphs.
    
    while time.time() < deadline:
        # Random permutation
        perm = np.random.permutation(n_vecs)
        
        selected = []
        selected_set = set()
        
        for i in perm:
            # Check if i conflicts with any selected
            # We can maintain a "blocked" set
            # If i is not in blocked, add it
            
            # Simple check:
            # conflict_mask[i] gives a boolean array of conflicts
            # We need to check if any selected index is in that array
            
            # Efficient way:
            # Maintain a bitset or just check intersection
            # Since N is small, simple loop over selected might be slow if selected is large.
            # But we can update a 'forbidden' set.
            
            # Let's use a boolean array 'is_forbidden'
            # When we select i, we mark all its neighbors as forbidden.
            
            pass # Placeholder for logic below

    # Re-implementing the loop properly
    best_set_size = 0
    best_indices = []
    
    # Pre-compute neighbors list for faster lookup?
    # neighbors[i] = list of j such that conflict_mask[i,j] is True
    
    neighbors = [np.where(conflict_mask[i])[0] for i in range(n_vecs)]
    
    while time.time() < deadline:
        perm = np.random.permutation(n_vecs)
        selected = []
        forbidden = np.zeros(n_vecs, dtype=bool)
        
        for i in perm:
            if not forbidden[i]:
                selected.append(i)
                # Mark neighbors as forbidden
                # forbidden[neighbors[i]] = True
                # Using numpy slicing might be faster
                forbidden[neighbors[i]] = True
                
        if len(selected) > best_set_size:
            best_set_size = len(selected)
            best_indices = selected
            
            # If we beat the seed, we can return early? 
            # Or keep searching for max?
            # The task asks for the max k.
            # If we find 221, we should try to find 222.
            # But if we find 1000, we are happy.
            
    # Construct the result array
    if best_indices:
        result = V[best_indices]
    else:
        # Fallback to seed if something went wrong
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        result = np.array(vectors, dtype=np.int64)
        
    return result
# EVOLVE-BLOCK-END
