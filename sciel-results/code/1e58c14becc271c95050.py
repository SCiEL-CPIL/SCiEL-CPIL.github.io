import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Generate all integer vectors with norm squared = 4
    # Type 1: +/- 2 e_i
    # Type 2: Four +/- 1s
    
    vectors = []
    
    # Type 1
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            vectors.append(v)
            
    # Type 2
    for indices in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(indices, signs):
                v[idx] = sign
            vectors.append(v)
            
    vectors = np.array(vectors, dtype=np.int64)
    N = len(vectors)
    
    # Compute dot products
    # dot[i, j] = vectors[i] . vectors[j]
    # We need to select a subset S such that for all i, j in S, i != j, dot[i, j] <= 2
    
    # Precompute conflict list: for each i, list of j such that dot[i, j] > 2
    # Note: dot[i, i] = 4, so i is always in conflict with itself.
    
    # Efficient dot product computation
    # N is ~5300. N^2 is ~28M.
    # Dot product matrix can be computed via matrix multiplication.
    dots = vectors @ vectors.T
    
    # conflicts[i] is a set/list of indices j where dots[i, j] > 2
    # We only care about j != i for the selection logic, but including i is fine if we check j != i or handle it.
    # Actually, if we select i, we cannot select any j in conflicts[i].
    # Since dots[i, i] = 4 > 2, i is in conflicts[i].
    
    # To speed up greedy, we can just check if any selected vector is in the conflict list of the candidate.
    # Let's store conflicts as lists of integers.
    # Using numpy argwhere might be slow for list conversion.
    # Let's just use the dots matrix directly in the loop?
    # No, N=5300, checking 5300 values per candidate is 5300*5300 = 28M ops per full scan.
    # If we do 100 restarts, 2.8B ops. Might be too slow in Python.
    
    # Better: Precompute conflict masks or lists.
    # Let's create a list of lists.
    conflicts = []
    for i in range(N):
        # Find j where dots[i, j] > 2
        # This is a row scan.
        # We can use np.where
        mask = dots[i] > 2
        conflicts.append(np.where(mask)[0])
        
    # Greedy selection with random restarts
    best_size = 0
    best_indices = []
    
    # Number of restarts
    # We have ~140s.
    # Each restart: shuffle N items, iterate N times.
    # Inside, check if candidate is compatible.
    # Compatibility check: check if any selected index is in conflicts[candidate].
    # If we maintain a set of selected indices, we can check:
    # for j in conflicts[candidate]:
    #     if j in selected_set: break
    
    # Average conflict degree is ~57.
    # So check is ~57 ops.
    # Total per restart: N * 57 ~ 300k ops.
    # 1000 restarts: 300M ops. Should be fast enough in NumPy/Python?
    # Python loop overhead might be significant.
    # Let's try 500 restarts.
    
    rng = np.random.default_rng(42)
    
    for restart in range(500):
        if time.time() > deadline:
            break
            
        order = rng.permutation(N)
        selected = []
        selected_set = set()
        
        for idx in order:
            # Check if idx conflicts with any selected
            # conflicts[idx] contains all j with dot > 2
            # We need to check if any j in selected_set is in conflicts[idx]
            # Or simpler: check if any j in conflicts[idx] is in selected_set
            
            compatible = True
            for j in conflicts[idx]:
                if j in selected_set:
                    compatible = False
                    break
            
            if compatible:
                selected.append(idx)
                selected_set.add(idx)
                
        if len(selected) > best_size:
            best_size = len(selected)
            best_indices = selected
            
    if best_size == 0:
        # Fallback to seed solution if something went wrong or no vectors found
        vectors_seed = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors_seed.append(v)
        return np.array(vectors_seed, dtype=np.int64)
        
    return vectors[best_indices]

# EVOLVE-BLOCK-END
