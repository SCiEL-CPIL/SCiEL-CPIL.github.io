"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product, combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Baseline: weight-2 vectors
    base_vectors = []
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    base_vectors.append(v)
    
    base_set = np.array(base_vectors, dtype=np.int64)
    best_set = base_set
    best_score = len(base_set)
    
    # Strategy: Maximize set of weight-4 vectors (norm 2).
    # Constraint: pairwise dot <= 2.
    # Candidates: All vectors with 4 non-zero entries from {1, -1}.
    # Count: C(11,4) * 2^4 = 330 * 16 = 5280.
    
    # Generate candidates
    candidates = []
    for support in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)
    
    cand_arr = np.array(candidates, dtype=np.int64)
    n_cands = len(cand_arr)
    
    # Greedy Independent Set with Randomization
    rng = np.random.default_rng(42)
    
    # Number of restarts depends on time. Let's try up to 100 restarts.
    max_restarts = 100
    
    for r in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        # Shuffle candidate indices
        indices = np.arange(n_cands)
        rng.shuffle(indices)
        
        # Maintain selected vectors as a numpy array for fast dot product checks
        # Start with empty array of shape (0, D)
        S = np.empty((0, D), dtype=np.int64)
        
        for idx in indices:
            if time.time() > deadline - 5:
                break
                
            v = cand_arr[idx]
            
            if S.shape[0] == 0:
                S = v[None, :].copy()
                continue
            
            # Check constraint: dot <= 2 for all selected
            dots = S @ v
            if np.all(dots <= 2):
                S = np.vstack((S, v[None, :]))
        
        if S.shape[0] > best_score:
            best_score = S.shape[0]
            best_set = S.copy()
            
    return best_set

# EVOLVE-BLOCK-END
