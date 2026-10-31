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
            
    # Type 4: three ±1's, rest 0 (norm^2 = 3)
    # Note: Including norm^2=3 vectors requires checking distances against max_norm_sq=4
    # dist_sq >= 4.
    # If u has norm sq 3, v has norm sq 4: 3+4-2dot >= 4 => dot <= 1.5 => dot <= 1.
    # If u has norm sq 3, v has norm sq 3: 3+3-2dot >= 4 => 2 >= 2dot => dot <= 1.
    # If u has norm sq 3, v has norm sq 2: 3+2-2dot >= 4 => 1 >= 2dot => dot <= 0.5 => dot <= 0.
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
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
    # (and i != j)
    conflict = (dist_sq < max_norm_sq)
    np.fill_diagonal(conflict, False)
    
    # Store conflict neighbors as lists for efficient access
    conflict_neighbors = [np.where(conflict[i])[0] for i in range(n)]
    
    # Deadline guard
    deadline = time.time() + 130  # 130s out of 150s limit
    
    # Greedy selection with multiple strategies and restarts
    best_selected = []
    best_size = 0
    
    def greedy_select(seed, strategy='random'):
        """Corrected Greedy independent set selection."""
        rng = np.random.RandomState(seed)
        remaining = set(range(n))
        selected = []
        
        # For 'min_conflict', we would need dynamic degrees. 
        # For simplicity and speed, we use 'random' or 'min_original_degree'.
        # Let's use 'random' as it is robust with restarts.
        
        while remaining:
            if time.time() > deadline:
                break
            
            rem_list = list(remaining)
            
            if strategy == 'random':
                chosen_local = rng.randint(len(rem_list))
            else:
                # Fallback to random if strategy not implemented
                chosen_local = rng.randint(len(rem_list))
            
            chosen_global = rem_list[chosen_local]
            selected.append(chosen_global)
            
            # FIX: Remove chosen node and all its neighbors from remaining
            # This ensures the selected set remains an independent set
            to_remove = set(conflict_neighbors[chosen_global])
            to_remove.add(chosen_global)
            remaining -= to_remove
            
        return selected
    
    def try_add_more(selected):
        """Try to add more vertices that don't conflict with selected set."""
        selected_set = set(selected)
        # Count how many selected neighbors each vertex has
        sel_neighbor_count = np.zeros(n, dtype=np.int32)
        for s in selected:
            for c in conflict_neighbors[s]:
                sel_neighbor_count[c] += 1
        
        added = []
        for i in range(n):
            if i not in selected_set and sel_neighbor_count[i] == 0:
                selected_set.add(i)
                added.append(i)
        return selected + added
    
    # Multiple restarts with random strategy (corrected greedy)
    # We can increase restarts since the greedy is faster (no count updates)
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
        
        seed = 42 + attempt * 7919
        strategy = 'random'
        
        selected = greedy_select(seed, strategy)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected
    
    if not best_selected:
        # Fallback: return empty or single vector if something went wrong
        return np.zeros((0, D), dtype=np.int64)
        
    result = candidates[best_selected]
    return result
# EVOLVE-BLOCK-END
