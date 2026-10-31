"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    Strategy:
    1. Generate a large candidate pool of short integer vectors (norm^2 = 2 and 4).
    2. Use a two-phase selection strategy:
       - Phase 1: Find a large set of norm^2=4 vectors (Type 2 and 3) that are mutually
         at distance >= 2 (dist_sq >= 4). This sets the "scale" of the configuration.
       - Phase 2: Try to add norm^2=2 vectors (Type 1) that are at distance >= 2 from
         all selected vectors. This is beneficial because adding norm^2=2 vectors doesn't
         increase the max norm (which remains 2 if any norm^2=4 vector is present), 
         but allows denser packing if the norm^2=2 vectors are orthogonal to the norm^2=4 ones.
       
       Actually, if we have any norm^2=4 vector, max_norm = 2. So ALL pairs must have dist_sq >= 4.
       This means we just need to find the maximum independent set in the conflict graph
       where conflict is dist_sq < 4.
       
       However, the current seed only uses Type 1 vectors (norm^2=2), giving 220 vectors.
       The inspirations use a mixed pool and greedy MIS, getting ~2.57x improvement.
       
       Let's use the inspiration's approach: mixed pool + greedy MIS with multiple restarts.
       But we can improve the greedy by using a smarter ordering.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Generate candidates
    candidates = []
    
    # Type 1: two ±1's, rest 0 (norm^2 = 2)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
    
    # Type 2: one ±2, rest 0 (norm^2 = 4)
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates.append(v)
    
    # Type 3: four ±1's, rest 0 (norm^2 = 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    # Check time
    if time.time() > deadline:
        # Fallback: return just Type 1 vectors
        type1 = candidates[:220]
        return type1
    
    # Compute norms squared
    norms_sq = np.sum(candidates**2, axis=1)
    
    # Compute distance squared matrix
    dots = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    
    # Conflict: dist_sq < 4 (we need dist_sq >= 4 for validity when max_norm = 2)
    conflict = (dist_sq < 4)
    np.fill_diagonal(conflict, False)
    
    # Greedy MIS with multiple restarts
    best_selected = []
    best_size = 0
    np.random.seed(42)
    
    num_restarts = 100
    
    for attempt in range(num_restarts):
        if time.time() > deadline:
            break
        
        remaining = np.ones(n, dtype=bool)
        degrees = conflict.sum(axis=1).copy()
        selected = []
        
        while remaining.any():
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break
            
            rem_degrees = degrees[rem_indices]
            min_deg = np.min(rem_degrees)
            min_mask = rem_degrees == min_deg
            min_indices_local = np.where(min_mask)[0]
            
            chosen_local = np.random.choice(min_indices_local)
            chosen_global = rem_indices[chosen_local]
            
            selected.append(chosen_global)
            
            neigh_mask = conflict[chosen_global]
            
            to_remove_mask = np.zeros(n, dtype=bool)
            to_remove_mask[chosen_global] = True
            to_remove_mask[neigh_mask] = True
            
            actual_remove = to_remove_mask & remaining
            
            if np.any(actual_remove):
                deg_decrease = conflict[actual_remove][:, remaining].sum(axis=0)
                degrees[remaining] -= deg_decrease
            
            remaining[actual_remove] = False
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected[:]
    
    if not best_selected:
        # Fallback
        return candidates[:220]
    
    result = candidates[best_selected]
    
    # Verify the result
    if len(result) > 1:
        r_dots = result @ result.T
        r_norms_sq = np.sum(result**2, axis=1)
        r_dist_sq = r_norms_sq[:, None] + r_norms_sq[None, :] - 2 * r_dots
        np.fill_diagonal(r_dist_sq, np.inf)
        min_dist_sq = np.min(r_dist_sq)
        max_norm = np.max(r_norms_sq)
        # Condition: min_dist_sq >= max_norm
        if min_dist_sq < max_norm - 1e-9:
            # Invalid, fallback to Type 1 only
            return candidates[:220]
    
    return result

# EVOLVE-BLOCK-END
