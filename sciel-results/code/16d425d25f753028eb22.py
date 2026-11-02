import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Generate candidates with squared norm 4
    candidates = []
    
    # Type 1: One 2, rest 0: ±2e_i
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four 1s, rest 0: ±e_i ± e_j ± e_k ± e_l
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # Conflict: two vectors conflict if their distance < 2, i.e., |v_i - v_j|^2 < 4
    # |v_i - v_j|^2 = |v_i|^2 + |v_j|^2 - 2*v_i·v_j = 4 + 4 - 2*M[i,j] = 8 - 2*M[i,j]
    # Need distance >= 2, so |v_i - v_j|^2 >= 4, so 8 - 2*M[i,j] >= 4, M[i,j] <= 2
    # Conflict if M[i,j] > 2
    conflict = (M > 2).astype(np.int8)
    np.fill_diagonal(conflict, 0)
    
    best_set = None
    best_count = 0
    
    rng = np.random.default_rng(42)
    
    # Tomita-style Branch and Bound for Maximum Independent Set
    # Using coloring bound for pruning
    def tomita_bb(order, conflict_mat, N):
        """Branch and bound for MIS using coloring bound."""
        best = []
        best_size = 0
        
        def color_bound(vertices):
            """Greedy coloring to get upper bound on independent set size."""
            if len(vertices) == 0:
                return 0
            uncolored = list(vertices)
            color_classes = []
            while uncolored:
                current_color = []
                i = 0
                while i < len(uncolored):
                    v = uncolored[i]
                    # Check if v conflicts with any vertex already in current_color
                    can_add = True
                    for cv in current_color:
                        if conflict_mat[v, cv]:
                            can_add = False
                            break
                    if can_add:
                        current_color.append(v)
                        uncolored.pop(i)
                    else:
                        i += 1
                color_classes.append(current_color)
            return len(color_classes)
        
        def expand(vertices, current_set):
            nonlocal best, best_size
            if not vertices:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best = list(current_set)
                return
            if len(current_set) + color_bound(vertices) <= best_size:
                return
            
            # Pick a vertex with maximum degree in the subgraph
            max_deg = -1
            chosen = -1
            for v in vertices:
                deg = sum(1 for u in vertices if conflict_mat[v, u])
                if deg > max_deg:
                    max_deg = deg
                    chosen = v
            
            # Branch: include chosen
            current_set.append(chosen)
            remaining = [v for v in vertices if v != chosen and not conflict_mat[chosen, v]]
            expand(remaining, current_set)
            current_set.pop()
            
            # Branch: exclude chosen
            remaining = [v for v in vertices if v != chosen]
            expand(remaining, current_set)
        
        expand(list(order), [])
        return best, best_size
    
    # First get a good initial solution with greedy
    def greedy_select(order, conflict_mat, N):
        selected_orig = np.zeros(N, dtype=bool)
        count = 0
        for idx in order:
            if count == 0:
                selected_orig[idx] = True
                count += 1
            else:
                if not np.any(conflict_mat[idx] & selected_orig):
                    selected_orig[idx] = True
                    count += 1
        return selected_orig, count
    
    # Strategy A: Get initial solution via multiple greedy passes
    n_trials = 500
    for trial in range(n_trials):
        if time.time() > deadline - 30:
            break
        
        if trial == 0:
            order = np.arange(N)
        else:
            order = rng.permutation(N)
        
        selected_mask, count = greedy_select(order, conflict, N)
        
        if count > best_count:
            best_count = count
            best_set = selected_mask.copy()
    
    # Strategy B: Use Tomita BB on a reduced graph
    # Take the best greedy solution and try to improve it
    if best_set is not None and time.time() < deadline - 10:
        selected_indices = np.where(best_set)[0]
        
        # Try to find a better solution using BB on unselected vertices
        # that are compatible with the current selection
        unselected = np.where(~best_set)[0]
        
        # Filter to only those compatible with current set
        compatible_unselected = []
        for idx in unselected:
            if not np.any(conflict[idx] & best_set):
                compatible_unselected.append(idx)
        
        if compatible_unselected:
            # Try to add more via BB on the compatible subgraph
            # For efficiency, just try greedy on compatible ones first
            compatible_array = np.array(compatible_unselected)
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                order = compatible_array[rng.permutation(len(compatible_array))]
                added = 0
                for idx in order:
                    if not np.any(conflict[idx] & best_set):
                        best_set[idx] = True
                        best_count += 1
                        added += 1
                if added == 0:
                    break
    
    # Strategy C: Local search improvement
    if best_set is not None and time.time() < deadline - 5:
        for ls_iter in range(50):
            if time.time() > deadline - 3:
                break
            
            selected_indices = np.where(best_set)[0]
            if len(selected_indices) < 5:
                break
            
            # Try removing 2-3 random selected and re-adding greedily
            n_remove = min(3, len(selected_indices))
            remove_candidates = rng.choice(len(selected_indices), size=n_remove, replace=False)
            remove_set = selected_indices[remove_candidates]
            
            test_set = best_set.copy()
            for r in remove_set:
                test_set[r] = False
            test_count = best_count - n_remove
            
            # Greedily add from unselected, prioritizing low-conflict
            unselected_test = np.where(~test_set)[0]
            if len(unselected_test) > 0:
                # Vectorized conflict count
                conf_counts = np.array([np.sum(conflict[u] & test_set) for u in unselected_test])
                order_add = unselected_test[np.argsort(conf_counts)]
                
                for add_idx in order_add:
                    if time.time() > deadline - 3:
                        break
                    if not np.any(conflict[add_idx] & test_set):
                        test_set[add_idx] = True
                        test_count += 1
            
            if test_count > best_count:
                best_set = test_set
                best_count = test_count
    
    # Strategy D: Final sweep - add any remaining compatible vectors
    if best_set is not None and time.time() < deadline - 1:
        unselected = np.where(~best_set)[0]
        for idx in unselected:
            if time.time() > deadline - 0.5:
                break
            if not np.any(conflict[idx] & best_set):
                best_set[idx] = True
                best_count += 1
    
    if best_set is None:
        vectors = []
        for i in range(D):
            for s in [1, -1]:
                v = [0] * D
                v[i] = 2 * s
                vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return candidates[best_set].astype(np.int64)
