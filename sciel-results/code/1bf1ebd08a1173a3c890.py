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
    
    Uses vectorized greedy with improved local search on candidate vectors
    with norm^2 = 2 and 4.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
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
    
    # Pre-compute initial degrees
    initial_degrees = conflict.sum(axis=1).astype(np.int32)
    
    def greedy_select(seed):
        """Greedy selection with minimum degree heuristic."""
        rng = np.random.RandomState(seed)
        remaining = np.ones(n, dtype=bool)
        degrees = initial_degrees.copy()
        selected = []
        
        while remaining.any():
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break
            
            rem_degrees = degrees[rem_indices]
            min_deg = np.min(rem_degrees)
            min_mask = rem_degrees == min_deg
            min_indices_local = np.where(min_mask)[0]
            
            chosen_local = rng.choice(min_indices_local)
            chosen_global = rem_indices[chosen_local]
            
            selected.append(chosen_global)
            
            # Remove chosen and its neighbors from remaining
            neigh_mask = conflict[chosen_global]
            to_remove_mask = neigh_mask.copy()
            to_remove_mask[chosen_global] = True
            
            actual_remove = to_remove_mask & remaining
            
            if np.any(actual_remove):
                # Update degrees for remaining vertices
                removed_indices = np.where(actual_remove)[0]
                still_remaining = np.where(remaining)[0]
                if len(still_remaining) > 0 and len(removed_indices) > 0:
                    deg_dec = conflict[np.ix_(removed_indices, still_remaining)].sum(axis=0)
                    degrees[still_remaining] -= deg_dec
            
            remaining[actual_remove] = False
        
        return selected
    
    def local_search(selected, max_iters=100):
        """Try to improve selection via swap moves."""
        selected_set = set(selected)
        remaining_set = set(range(n)) - selected_set
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            if time.time() > deadline:
                break
            
            # Try systematic swaps: for each selected node, try swapping with each remaining node
            sel_list = list(selected_set)
            rem_list = list(remaining_set)
            
            if not sel_list or not rem_list:
                break
            
            # Try random swaps more aggressively
            n_swaps_tried = 0
            max_swaps_per_iter = min(50, len(sel_list) * len(rem_list))
            
            while n_swaps_tried < max_swaps_per_iter:
                n_swaps_tried += 1
                if time.time() > deadline:
                    break
                    
                s = sel_list[np.random.randint(len(sel_list))]
                r = rem_list[np.random.randint(len(rem_list))]
                
                # Check if removing s and adding r is valid
                # r must not conflict with any selected node except s
                sel_arr_check = np.array([x for x in selected_set if x != s])
                if len(sel_arr_check) > 0:
                    if np.any(conflict[r, sel_arr_check]):
                        continue
                
                # Valid swap!
                selected_set.remove(s)
                selected_set.add(r)
                remaining_set.remove(r)
                remaining_set.add(s)
                improved = True
                break
        
        return list(selected_set)
    
    # Run multiple greedy restarts with local search
    np.random.seed(42)
    best_selected = []
    best_size = 0
    
    # Number of restarts - we'll adjust based on time budget
    max_restarts = 500
    
    for attempt in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Greedy phase
        selected = greedy_select(attempt * 7919 + 42)
        
        # Local search phase
        if time.time() < deadline - 10:
            selected = local_search(selected, max_iters=50)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected.copy()
    
    if not best_selected:
        # Fallback: return a single vector
        return candidates[:1]
    
    result = candidates[best_selected]
    return result
# EVOLVE-BLOCK-END
