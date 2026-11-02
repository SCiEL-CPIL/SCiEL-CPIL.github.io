import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START
def generate_candidates_norm5():
    """Generate all integer vectors in R^11 with norm squared 5."""
    cands = []
    
    # Type 1: Permutations of (2, 1, 0, ..., 0) with signs
    # 2^2 + 1^2 = 5
    # Number of such vectors: P(11,2) * 4 = 110 * 4 = 440
    for i, j in itertools.permutations(range(D), 2):
        for s1 in (1, -1):
            for s2 in (1, -1):
                v = [0] * D
                v[i] = 2 * s1
                v[j] = 1 * s2
                cands.append(v)
                
    # Type 2: Permutations of (1, 1, 1, 1, 1, 0, ..., 0) with signs
    # 1^2 * 5 = 5
    # Number of such vectors: C(11,5) * 32 = 462 * 32 = 14784
    for indices in itertools.combinations(range(D), 5):
        for signs in itertools.product([1, -1], repeat=5):
            v = [0] * D
            for idx, s in zip(indices, signs):
                v[idx] = s
            cands.append(v)
            
    return np.array(cands, dtype=np.int64)

def construct():
    start_time = time.time()
    deadline = 100.0  # Keep safety margin from 120s limit
    
    # Generate candidates
    candidates = generate_candidates_norm5()
    
    # Shuffle to avoid bias in greedy selection
    np.random.shuffle(candidates)
    
    selected = []
    
    # Greedy selection
    # We want max size set S such that min_{u!=v in S} ||u-v||^2 >= 5
    for v in candidates:
        if time.time() > deadline:
            break
            
        if not selected:
            selected.append(v)
            continue
            
        # Convert to array for vectorized distance calculation
        # This is O(K*D) per step
        S_arr = np.array(selected)
        
        # Calculate squared distances from v to all vectors in S
        # (S_arr - v) ** 2
        dists = np.sum((S_arr - v) ** 2, axis=1)
        
        # Check if minimum distance is at least sqrt(5) (squared: 5)
        if np.min(dists) >= 5:
            selected.append(v)
            
    if not selected:
        return np.array([], dtype=np.int64)
        
    return np.array(selected, dtype=np.int64)
# EVOLVE-BLOCK-END
