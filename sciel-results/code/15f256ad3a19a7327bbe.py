import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Step 1: Generate all 4-subsets
    subsets = list(itertools.combinations(range(D), 4))
    n_subsets = len(subsets) # 330
    
    # Precompute intersection sizes between all pairs of subsets
    # inter[i][j] = len(subsets[i] & subsets[j])
    # We only care if inter[i][j] >= 3
    # For efficiency, we can store the list of incompatible subsets for each subset
    incompatible_supports = [set() for _ in range(n_subsets)]
    for i in range(n_subsets):
        for j in range(i + 1, n_subsets):
            if len(set(subsets[i]) & set(subsets[j])) >= 3:
                incompatible_supports[i].add(j)
                incompatible_supports[j].add(i)
                
    # Step 2: Find a large independent set of supports
    # Greedy with randomization
    best_supports = []
    best_size = 0
    
    for seed in range(1000):
        if time.time() > deadline - 10:
            break
        rng = random.Random(seed)
        order = list(range(n_subsets))
        rng.shuffle(order)
        
        selected = set()
        for idx in order:
            # Check if idx is compatible with all selected
            # Compatible means no intersection >= 3
            # i.e., idx is not in incompatible_supports[s] for any s in selected
            # Faster: check if selected & incompatible_supports[idx] is empty
            if not (selected & incompatible_supports[idx]):
                selected.add(idx)
                
        if len(selected) > best_size:
            best_size = len(selected)
            best_supports = list(selected)
            
    # Step 3: Construct vectors
    vectors = []
    
    # Add weight 4 vectors for selected supports
    for idx in best_supports:
        subset = subsets[idx]
        # Generate all 16 sign combinations
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for i, pos in enumerate(subset):
                v[pos] = signs[i]
            vectors.append(v)
            
    # Add weight 1 vectors
    for pos in range(D):
        for sign in [2, -2]:
            v = np.zeros(D, dtype=np.int64)
            v[pos] = sign
            vectors.append(v)
            
    # Convert to array
    V = np.array(vectors, dtype=np.int64)
    
    return V
