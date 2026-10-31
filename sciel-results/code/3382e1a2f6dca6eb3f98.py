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
    
    # Store as list of lists of conflict neighbors for efficient access
    conflict_neighbors = [np.where(conflict[i])[0] for i in range(n)]
    
    # Deadline guard
    deadline = time.time() + 130  # 130s out of 150s limit
    
    # Greedy selection with multiple strategies and restarts
    best_selected = []
    best_size = 0
    
    def greedy_select(seed, strategy='min_conflict'):
        """Greedy independent set selection with different strategies."""
        rng = np.random.RandomState(seed)
        remaining = set(range(n))
        selected = []
        
        # Precompute initial conflict counts
        conflict_counts = np.array([len(conflict_neighbors[i]) for i in range(n)], dtype=np.int32)
        
        while remaining:
            if time.time() > deadline:
                break
            
            rem_list = list(remaining)
            rem_array = np.array(rem_list)
            
            # Get conflict counts for remaining nodes
            rem_conflict_counts = conflict_counts[rem_array]
            
            if strategy == 'min_conflict':
                min_conf = np.min(rem_conflict_counts)
                min_indices = np.where(rem_conflict_counts == min_conf)[0]
                chosen_local = rng.choice(min_indices)
            elif strategy == 'random':
                chosen_local = rng.randint(len(rem_list))
            else:
                min_conf = np.min(rem_conflict_counts)
                min_indices = np.where(rem_conflict_counts == min_conf)[0]
                chosen_local = rng.choice(min_indices)
            
            chosen_global = rem_list[chosen_local]
            selected.append(chosen_global)
            remaining.remove(chosen_global)
            
            # Update conflict counts for remaining nodes
            for c in conflict_neighbors[chosen_global]:
                if c in remaining:
                    conflict_counts[c] -= 1
            conflict_counts[chosen_global] = 0
        
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
    
    # Multiple restarts with different strategies
    for attempt in range(100):
        if time.time() > deadline - 5:
            break
        
        seed = 42 + attempt * 7919
        # Alternate between strategies
        if attempt % 5 == 0:
            strategy = 'random'
        else:
            strategy = 'min_conflict'
        
        selected = greedy_select(seed, strategy)
        
        # Try to add more vertices
        selected = try_add_more(selected)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected
    
    result = candidates[best_selected]
    return result
# EVOLVE-BLOCK-END
