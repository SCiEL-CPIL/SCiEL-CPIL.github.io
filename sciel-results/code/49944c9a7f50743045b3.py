import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    def greedy_select(pool, max_dot=2):
        n = len(pool)
        selected = []
        
        for i in range(n):
            if time.time() > deadline:
                break
            v = pool[i]
            
            if not selected:
                selected.append(v)
                continue
            
            dots = np.array(selected) @ v
            if np.max(dots) <= max_dot:
                selected.append(v)
        
        return np.array(selected, dtype=np.int64)
    
    # Generate weight-2 vectors (norm = 2)
    pool2 = []
    for comb in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for idx, sign in zip(comb, signs):
                v[idx] = sign
            pool2.append(v)
    
    # Generate weight-4 vectors (norm = 2)
    pool4 = []
    for comb in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(comb, signs):
                v[idx] = sign
            pool4.append(v)
    
    # Combine: try weight-2 first (they're all mutually compatible), then add weight-4
    # Or just combine all and do greedy
    
    all_pool = np.array(pool2 + pool4, dtype=np.int64)
    n_pool = len(all_pool)
    
    best_set = None
    best_len = 0
    
    # The weight-2 vectors are all mutually compatible (max dot = 2)
    # So we can start with all weight-2, then greedily add weight-4
    w2 = np.array(pool2, dtype=np.int64)
    w4 = np.array(pool4, dtype=np.int64)
    
    # Start with all weight-2
    selected = [w2]
    
    # Add weight-4 vectors that are compatible with all weight-2
    for v in w4:
        if time.time() > deadline:
            break
        dots_w2 = w2 @ v
        if np.max(dots_w2) <= 2:
            # Check against already selected weight-4
            if len(selected) > 1:
                dots_w4 = np.array(selected[1:]) @ v
                if np.max(dots_w4) <= 2:
                    selected.append(v)
            else:
                selected.append(v)
    
    result = np.vstack(selected)
    best_len = len(result)
    best_set = result
    
    # Try random shuffles of the combined pool for potentially better results
    if time.time() < deadline - 10:
        for trial in range(15):
            if time.time() > deadline - 5:
                break
            rng = np.random.default_rng(trial * 31337)
            order = rng.permutation(n_pool)
            ordered_pool = all_pool[order]
            result = greedy_select(ordered_pool, max_dot=2)
            if len(result) > best_len:
                best_len = len(result)
                best_set = result
    
    if best_set is None or len(best_set) == 0:
        return np.array([[1,1,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    
    return best_set
# EVOLVE-BLOCK-END
