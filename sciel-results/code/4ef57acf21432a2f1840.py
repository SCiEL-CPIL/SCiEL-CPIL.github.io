import numpy as np
import itertools

def construct():
    D = 11
    
    # Generate type-1 vectors: one ±2, rest 0 (norm^2 = 4)
    type1 = []
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * s
            type1.append(v)
    type1 = np.array(type1, dtype=np.int64)
    
    # Generate type-2 vectors: four ±1's, rest 0 (norm^2 = 4)
    type2 = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            type2.append(v)
    type2 = np.array(type2, dtype=np.int64)
    
    n2 = len(type2)
    
    # Compute conflict matrix for type-2 vectors only
    # Two type-2 vectors conflict if ||u-v||^2 < 4
    # ||u-v||^2 = 4 + 4 - 2<u,v> = 8 - 2<u,v>
    # Conflict if 8 - 2<u,v> < 4 → <u,v> > 2 → <u,v> >= 3
    
    dots2 = type2 @ type2.T
    conflict2 = (dots2 > 2)
    np.fill_diagonal(conflict2, False)
    
    # Greedy maximum independent set on type-2 conflict graph
    np.random.seed(42)
    best_selected2 = []
    best_size2 = 0
    
    for attempt in range(100):
        remaining = set(range(n2))
        selected = []
        
        while remaining:
            rem_list = list(remaining)
            rem_array = np.array(rem_list)
            
            sub_conflict = conflict2[np.ix_(rem_array, rem_array)]
            conflict_counts = sub_conflict.sum(axis=1)
            
            min_conf = np.min(conflict_counts)
            min_indices = np.where(conflict_counts == min_conf)[0]
            chosen_local = np.random.choice(min_indices)
            chosen_global = rem_list[chosen_local]
            
            selected.append(chosen_global)
            remaining.remove(chosen_global)
            
            conflicts_with_chosen = np.where(conflict2[chosen_global])[0]
            to_remove = set(conflicts_with_chosen.tolist()) & remaining
            remaining -= to_remove
        
        if len(selected) > best_size2:
            best_size2 = len(selected)
            best_selected2 = selected
    
    # Combine: all type-1 + selected type-2
    result = np.vstack([type1, type2[best_selected2]])
    return result
