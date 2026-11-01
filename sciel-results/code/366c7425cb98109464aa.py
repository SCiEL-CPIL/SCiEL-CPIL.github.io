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
    
    We use a greedy set-packing with future-conflict-radius heuristic on
    weight-4 vectors, plus local search.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
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
    
    def greedy_select_future_conflict(seed):
        """
        Greedy selection that prioritizes candidates with smallest 'future conflict radius'.
        Future conflict radius = number of remaining candidates that would be eliminated
        if we pick this candidate (i.e., its degree among remaining vertices).
        This preserves search space flexibility.
        """
        rng = np.random.RandomState(seed)
        remaining = np.ones(n, dtype=bool)
        degrees = initial_degrees.copy()
        selected = []
        
        while remaining.any():
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break
            
            rem_degrees = degrees[rem_indices]
            
            # Find candidates with minimum future conflict radius (degree)
            min_deg = np.min(rem_degrees)
            min_mask = rem_degrees == min_deg
            min_indices_local = np.where(min_mask)[0]
            
            # Among ties, break randomly
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
    
    def local_search(selected, max_iters=50):
        """Try to improve selection via swap moves."""
        selected_set = set(selected)
        remaining_set = set(range(n)) - selected_set
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            for _ in range(20):
                if time.time() > deadline:
                    break
                    
                if len(selected_set) == 0 or len(remaining_set) == 0:
                    break
                    
                sel_list = list(selected_set)
                rem_list = list(remaining_set)
                
                for _ in range(5):
                    s = sel_list[np.random.randint(len(sel_list))]
                    r = rem_list[np.random.randint(len(rem_list))]
                    
                    # Check: for all selected nodes except s, r must not conflict
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
                if improved:
                    break
                if time.time() > deadline:
                    break
        
        return list(selected_set)
    
    # Run multiple greedy restarts with local search
    np.random.seed(42)
    best_selected = []
    best_size = 0
    
    max_restarts = 200
    
    for attempt in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Greedy phase with future-conflict-radius heuristic
        selected = greedy_select_future_conflict(attempt * 7919 + 42)
        
        # Local search phase
        if time.time() < deadline - 5:
            selected = local_search(selected, max_iters=30)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected.copy()
    
    if not best_selected:
        return candidates[:1]
    
    result = candidates[best_selected]
    return result
# EVOLVE-BLOCK-END
