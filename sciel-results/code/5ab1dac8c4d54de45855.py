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
    
    We use a vectorized greedy approach on candidate vectors with norm^2 = 2 and 4.
    The key constraint: for vectors u, v with max_norm_sq = 4,
    we need ||u-v||^2 >= 4, i.e., ||u||^2 + ||v||^2 - 2<u,v> >= 4.
    
    For norm^2=2 vectors: 2+2-2<u,v> >= 4 → <u,v> <= 0
    For norm^2=4 vectors: 4+4-2<u,v> >= 4 → <u,v> <= 2
    For mixed (norm^2=2, norm^2=4): 2+4-2<u,v> >= 4 → <u,v> <= 1
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Generate candidate vectors
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
    
    # Compute squared norms
    norms_sq = np.sum(candidates**2, axis=1)
    max_norm_sq = 4  # Maximum possible norm^2 in our set
    
    # Compute pairwise dot products (vectorized)
    dots = candidates @ candidates.T
    
    # Compute squared distances
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    
    # Conflict matrix: conflict[i,j] = True if dist_sq[i,j] < max_norm_sq
    conflict = (dist_sq < max_norm_sq)
    np.fill_diagonal(conflict, False)
    
    # Precompute adjacency lists for faster local search
    adj = [np.where(conflict[i])[0] for i in range(n)]
    deg = np.array([len(adj[i]) for i in range(n)], dtype=np.int32)
    
    # Greedy selection with multiple restarts and improved local search
    best_selected = []
    best_size = 0
    np.random.seed(42)
    
    num_attempts = 500
    
    for attempt in range(num_attempts):
        if time.time() > deadline - 15.0:
            break
        
        remaining = np.ones(n, dtype=bool)
        degrees = deg.copy()
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
                # Update degrees efficiently
                for idx in np.where(actual_remove)[0]:
                    for nb in adj[idx]:
                        if remaining[nb]:
                            degrees[nb] -= 1
            
            remaining[actual_remove] = False
        
        # Improved local search: pairwise swap with precomputed adjacency
        selected_set = set(selected)
        improved = True
        max_local_iters = 50
        local_iter = 0
        while improved and local_iter < max_local_iters and time.time() < deadline - 5.0:
            improved = False
            local_iter += 1
            for i in range(len(selected)):
                if time.time() > deadline - 5.0:
                    break
                removed = selected[i]
                
                # Find candidates that could replace removed
                # They must not conflict with any other selected vertex
                # and must not be the removed vertex itself
                other_selected = [s for j, s in enumerate(selected) if j != i]
                
                # A candidate c is valid if:
                # 1. c is not in selected
                # 2. c does not conflict with any vertex in other_selected
                # 3. c is adjacent to removed (to potentially enable further swaps)
                # Actually, we just need c to not conflict with other_selected.
                
                # Build set of vertices that conflict with any other_selected
                blocked = set()
                for s in other_selected:
                    blocked.update(adj[s])
                blocked.update(selected_set)  # Can't select already-selected
                blocked.discard(removed)  # removed is not selected anymore
                
                # Find candidates adjacent to removed that are not blocked
                candidates_to_try = [c for c in adj[removed] if c not in blocked]
                
                if candidates_to_try:
                    # Pick the one with minimum degree
                    best_c = min(candidates_to_try, key=lambda c: degrees[c])
                    
                    # Verify it doesn't conflict with other_selected (should be guaranteed)
                    valid = True
                    for s in other_selected:
                        if conflict[best_c, s]:
                            valid = False
                            break
                    
                    if valid:
                        new_selected = list(selected)
                        new_selected[i] = best_c
                        if len(new_selected) > len(selected):
                            selected = new_selected
                            selected_set = set(selected)
                            improved = True
                            break
                        elif len(new_selected) == len(selected):
                            # Same size, accept if it reduces total degree
                            old_total_deg = sum(degrees[s] for s in selected)
                            new_total_deg = sum(degrees[s] for s in new_selected)
                            if new_total_deg < old_total_deg:
                                selected = new_selected
                                selected_set = set(selected)
                                improved = True
                                break
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = list(selected)
    
    result = candidates[best_selected]
    return result
# EVOLVE-BLOCK-END
