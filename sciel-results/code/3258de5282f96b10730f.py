"""Rewritten program: symmetry-reduced maximum independent set for kissing number in dimension 11."""
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    D = 11

    # Generate all weight-4 vectors with ±1 entries
    # C(11,4) = 330 positions, 2^4 = 16 sign combos -> 5280 vectors
    candidates_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)  # 5280

    # Precompute the full dot product matrix using int16 to avoid overflow
    dot = candidates.astype(np.int16) @ candidates.T.astype(np.int16)
    
    # Conflict matrix: C[i,j] = True if i and j CANNOT be selected together
    # Constraint for weight-4: dot <= 2
    # So if dot > 2, they conflict.
    conflict = (dot > 2).astype(np.uint8)
    np.fill_diagonal(conflict, 0)

    def min_degree_greedy(conflict_matrix, n, seed=None, randomize_ties=False):
        """
        Greedy independent set using minimum degree heuristic.
        """
        if seed is not None:
            np.random.seed(seed)
        
        active = np.ones(n, dtype=bool)
        selected = []
        
        for _ in range(n):
            if not active.any():
                break
            
            # Compute degrees of active nodes
            deg = conflict_matrix @ active
            deg[~active] = n + 1
            
            min_deg = np.min(deg)
            candidates_idx = np.where(deg == min_deg)[0]
            
            if randomize_ties and len(candidates_idx) > 1:
                chosen = candidates_idx[np.random.randint(len(candidates_idx))]
            else:
                chosen = candidates_idx[0]
            
            selected.append(chosen)
            
            active[chosen] = False
            active[conflict_matrix[chosen] == 1] = False
            
            if time.time() > deadline - 5.0:
                break

        return selected

    def local_search_improve(selected_indices, conflict_matrix, n, time_limit):
        """
        Improve the set by trying to add vertices or swap.
        """
        selected = set(selected_indices)
        sel_mask = np.zeros(n, dtype=bool)
        for idx in selected:
            sel_mask[idx] = True
            
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Try to add
            conflict_counts = conflict_matrix @ sel_mask
            addable = (~sel_mask) & (conflict_counts == 0)
            
            if addable.any():
                candidates_to_add = np.where(addable)[0]
                new_idx = candidates_to_add[0]
                
                selected.add(new_idx)
                sel_mask[new_idx] = True
                improved = True
                continue
            
            # 2. Try to swap
            if len(selected) > 10:
                s = np.random.choice(list(selected))
                selected.remove(s)
                sel_mask[s] = False
                
                conflict_counts = conflict_matrix @ sel_mask
                addable = (~sel_mask) & (conflict_counts == 0)
                
                if addable.any():
                    candidates_to_add = np.where(addable)[0]
                    new_idx = candidates_to_add[0]
                    selected.add(new_idx)
                    sel_mask[new_idx] = True
                    improved = True
                    continue
            
            break
            
        return list(selected)

    best_selected = []
    best_len = 0
    
    # Initial run
    sel = min_degree_greedy(conflict, n, seed=42, randomize_ties=False)
    best_selected = sel
    best_len = len(sel)
    
    # Run random tie-breaking trials
    trial = 0
    while time.time() < deadline - 10.0:
        trial += 1
        sel = min_degree_greedy(conflict, n, seed=trial, randomize_ties=True)
        
        if time.time() < deadline - 5.0:
            sel = local_search_improve(sel, conflict, n, deadline - 5.0)
        
        if len(sel) > best_len:
            best_len = len(sel)
            best_selected = sel
        
    if not best_selected:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)

    result = candidates[np.array(best_selected, dtype=np.int64)]
    return result.astype(np.int64)

