"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    Strategy:
    1. Type 1 vectors (norm^2=2, two +/-1s): All 220 are mutually compatible.
       Condition: min dist >= max norm.
       Max norm = sqrt(2). Min dist^2 = 2.
       Dist^2 = 4 - 2*dot. dot <= 1 for distinct vectors. 4-2(1)=2. OK.
       Score = 220.
       
    2. Type 3 vectors (norm^2=4, four +/-1s):
       Max norm = 2. Min dist^2 >= 4.
       Dist^2 = 8 - 2*dot. Need 8 - 2*dot >= 4 => dot <= 2.
       Conflicts if dot > 2.
       We find a large independent set in the conflict graph.
       
    We compare the best result from Type 3 search with the fixed 220 from Type 1.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # --- Generate Type 1 Vectors (Baseline) ---
    type1_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                type1_vectors.append(v)
    type1_array = np.array(type1_vectors, dtype=np.int64)
    best_result = type1_array.copy()
    best_size = len(best_result)
    
    # --- Generate Type 3 Vectors ---
    type3_vectors = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            type3_vectors.append(v)
    type3_array = np.array(type3_vectors, dtype=np.int64)
    n3 = len(type3_array)
    
    if n3 == 0:
        return best_result
        
    # Compute dot products for Type 3
    dots3 = type3_array @ type3_array.T
    
    # Conflict if dot > 2 (since norm^2=4, dist^2 = 8 - 2*dot >= 4 => dot <= 2)
    conflict3 = (dots3 > 2)
    np.fill_diagonal(conflict3, False)
    
    # Degrees
    degrees3 = conflict3.sum(axis=1)
    
    # Greedy Independent Set with Random Restart
    # We use a simple greedy: pick node with min degree, remove it and neighbors.
    # Repeat.
    
    def greedy_independent_set(conflict, degrees, n, rng):
        remaining = np.ones(n, dtype=bool)
        deg = degrees.copy()
        selected = []
        
        while remaining.any():
            # Find remaining nodes
            rem_idx = np.where(remaining)[0]
            if len(rem_idx) == 0:
                break
            
            # Get degrees of remaining nodes
            rem_deg = deg[rem_idx]
            
            # Pick node with min degree (random tie-break)
            min_deg = np.min(rem_deg)
            min_mask = rem_deg == min_deg
            candidates_local = np.where(min_mask)[0]
            
            chosen_local = rng.choice(candidates_local)
            chosen_global = rem_idx[chosen_local]
            
            selected.append(chosen_global)
            
            # Remove chosen and its neighbors
            to_remove = conflict[chosen_global] | (np.arange(n) == chosen_global)
            to_remove = to_remove & remaining
            
            if np.any(to_remove):
                removed_idx = np.where(to_remove)[0]
                # Update degrees of still remaining nodes
                still_rem = np.where(remaining)[0]
                if len(still_rem) > 0 and len(removed_idx) > 0:
                    # deg[still_rem] -= sum(conflict[removed_idx, still_rem], axis=0)
                    deg_dec = conflict[np.ix_(removed_idx, still_rem)].sum(axis=0)
                    deg[still_rem] -= deg_dec
            
            remaining[to_remove] = False
        
        return selected

    best_t3_selected = []
    best_t3_size = 0
    
    # Run multiple restarts
    # Time budget: ~100s. Each restart is fast (N=5280, sparse graph).
    # Let's try 100-500 restarts depending on speed.
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        rng = np.random.RandomState(attempts * 12345 + 678)
        
        # Simple greedy
        selected = greedy_independent_set(conflict3, degrees3, n3, rng)
        
        if len(selected) > best_t3_size:
            best_t3_size = len(selected)
            best_t3_selected = selected.copy()
            
        # If we already beat the baseline significantly, we can stop or continue to polish
        if best_t3_size > best_size + 50 and attempts > 10:
            # Optional: break early if we have a good lead
            # But let's keep going to maximize
            pass

    if best_t3_size > best_size:
        best_result = type3_array[best_t3_selected]
        best_size = best_t3_size
        
    return best_result
# EVOLVE-BLOCK-END
