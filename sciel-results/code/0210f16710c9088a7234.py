"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import numpy as np
import time
import random

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 115

    def elapsed_ok():
        return (time.time() - start_time) < DEADLINE

    candidates = []
    norms_sq = []

    def add_candidate(v, n_sq):
        candidates.append(v)
        norms_sq.append(n_sq)

    # Generate candidates for N=8
    # Type 1: (±2, ±2, 0, ...) Norm^2 = 8
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                add_candidate(v, 8)
                
    # Type 2: (±1, ..., ±1, 0, ...) with 8 ones. Norm^2 = 8
    for pos_set in itertools.combinations(range(D), 8):
        for mask in range(256):
            v = [0] * D
            for idx, pos in enumerate(pos_set):
                if mask & (1 << idx):
                    v[pos] = -1
                else:
                    v[pos] = 1
            add_candidate(v, 8)
            
    # Type 3: (±2, ±1, ..., ±1, 0, ...) with one 2 and four 1s. Norm^2 = 4+4=8
    for pos_2 in range(D):
        others = [p for p in range(D) if p != pos_2]
        for pos_set_1 in itertools.combinations(others, 4):
            for s2 in (1, -1):
                for mask in range(16):
                    v = [0] * D
                    v[pos_2] = 2 * s2
                    for idx, pos in enumerate(pos_set_1):
                        v[pos] = -1 if (mask & (1 << idx)) else 1
                    add_candidate(v, 8)

    # Generate candidates for N=10
    # Type 4: (±3, ±1, 0, ...) Norm^2 = 9+1=10
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 3 * si
                v[j] = 1 * sj
                add_candidate(v, 10)

    # Type 5: (±2, ±2, ±1, ±1, 0, ...) Norm^2 = 4+4+1+1=10
    for i, j, k, l in itertools.combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = [0] * D
                        v[i] = 2 * si
                        v[j] = 2 * sj
                        v[k] = 1 * sk
                        v[l] = 1 * sl
                        add_candidate(v, 10)

    # Type 6: (±2, ±1, ..., ±1, 0, ...) with one 2 and six 1s. Norm^2 = 4+6=10
    for pos_2 in range(D):
        others = [p for p in range(D) if p != pos_2]
        for pos_set_1 in itertools.combinations(others, 6):
            for s2 in (1, -1):
                for mask in range(64):
                    v = [0] * D
                    v[pos_2] = 2 * s2
                    for idx, pos in enumerate(pos_set_1):
                        v[pos] = -1 if (mask & (1 << idx)) else 1
                    add_candidate(v, 10)

    # Type 7: (±1, ..., ±1, 0, ...) with ten 1s. Norm^2 = 10
    for pos_set in itertools.combinations(range(D), 10):
        for mask in range(1024):
            v = [0] * D
            for idx, pos in enumerate(pos_set):
                if mask & (1 << idx):
                    v[pos] = -1
                else:
                    v[pos] = 1
            add_candidate(v, 10)

    candidates_arr = np.array(candidates, dtype=np.int64)
    norms_arr = np.array(norms_sq, dtype=np.int64)
    n_candidates = len(candidates_arr)
    
    best_arr = np.array([], dtype=np.int64).reshape(0, D)
    best_norms = np.array([], dtype=np.int64)
    
    random.seed(42)
    
    # Try multiple random orders
    # We can limit restarts if time is tight, but the greedy is fast enough for ~200k candidates?
    # 200k * 500 checks * 11 dims = 1.1e9 ops. Might be slow.
    # Let's try 5 restarts.
    
    for restart in range(5):
        if not elapsed_ok():
            break
            
        indices = np.arange(n_candidates)
        random.shuffle(indices)
        
        selected_indices = []
        current_arr = np.array([], dtype=np.int64).reshape(0, D)
        current_norms = np.array([], dtype=np.int64)
        
        for idx in indices:
            if not elapsed_ok():
                break
                
            v = candidates_arr[idx]
            n_v = norms_arr[idx]
            
            if len(current_arr) == 0:
                current_arr = np.array([v], dtype=np.int64)
                current_norms = np.array([n_v], dtype=np.int64)
                selected_indices.append(idx)
            else:
                # Check compatibility: dot(u, v) <= min(norm_u, norm_v) / 2
                dots = current_arr @ v
                # min_norms is the element-wise minimum of current_norms and n_v (scalar)
                # Since n_v is scalar, min(current_norms, n_v) is a vector
                min_norms = np.minimum(current_norms, n_v)
                thresholds = min_norms // 2  # Integer division is safe since norms are even? 
                # Norms are 8 or 10. 8/2=4, 10/2=5.
                # If min is 8, threshold 4. If min is 10, threshold 5.
                
                if np.all(dots <= thresholds):
                    current_arr = np.vstack([current_arr, v])
                    current_norms = np.append(current_norms, n_v)
                    selected_indices.append(idx)
                    
        if len(current_arr) > len(best_arr):
            best_arr = current_arr.copy()
            best_norms = current_norms.copy()
            if len(current_arr) >= 500:
                break

    return best_arr

# EVOLVE-BLOCK-END
