import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # 1. Generate Weight-2 vectors (Baseline)
    v2_list = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                v2_list.append(v)
    v2_array = np.array(v2_list, dtype=np.int64)
    best_vecs = v2_array
    best_score = len(best_vecs)
    
    # 2. Generate Weight-4 vectors
    v4_list = []
    for indices in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, idx in enumerate(indices):
                v[idx] = signs[k]
            v4_list.append(v)
    
    if not v4_list:
        return best_vecs
        
    v4_array = np.array(v4_list, dtype=np.int64)
    n = len(v4_array)
    
    # 3. Compute Conflict Matrix for Weight-4
    # Conflict if SharedIndexCount == 3 AND SameSignCount == 3
    
    # Support matrix (1 if non-zero)
    support = (v4_array != 0).astype(np.int8)
    M_shared = support @ support.T
    
    # Same sign count
    # Pos part
    pos = (v4_array == 1).astype(np.int8)
    neg = (v4_array == -1).astype(np.int8)
    
    M_pos = pos @ pos.T
    M_neg = neg @ neg.T
    M_same = M_pos + M_neg
    
    # Conflict condition
    conflicts = (M_shared == 3) & (M_same == 3)
    
    # Remove self-conflicts (diagonal)
    np.fill_diagonal(conflicts, False)
    
    # 4. Greedy Independent Set
    # We want to select a large subset of indices from v4_array such that no two selected are connected in 'conflicts'
    
    available = np.ones(n, dtype=bool)
    selected = []
    
    # Precompute degrees to speed up "min degree" selection?
    # Or just iterate.
    # With n=5280, and sparse graph, we can do simple greedy.
    
    # Optimization: Shuffle indices to break symmetry / randomize tie-breaking
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    # Simple Greedy:
    # Iterate through shuffled indices.
    # If node is available, select it.
    # Mark it and its neighbors as unavailable.
    
    # To improve quality, we can try "Min Degree" heuristic.
    # But simple greedy with randomization is often good enough for sparse graphs.
    # Let's try simple greedy first.
    
    for idx in indices:
        if time.time() > deadline:
            break
        if available[idx]:
            selected.append(idx)
            # Mark neighbors as unavailable
            neighbors = np.where(conflicts[idx, :])[0]
            available[neighbors] = False
            available[idx] = False
            
    if len(selected) > best_score:
        best_vecs = v4_array[selected]
        best_score = len(best_vecs)
        
    # 5. Check time and return
    if time.time() > deadline:
        return best_vecs
        
    # Optional: Try Weight-3 if Weight-4 didn't beat 220?
    # Weight-3 is smaller candidate pool, likely smaller result.
    # But let's stick to Weight-4 as it has higher potential.
    
    return best_vecs
