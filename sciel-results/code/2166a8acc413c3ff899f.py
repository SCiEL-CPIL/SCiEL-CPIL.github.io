import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # Generate all weight-4 vectors (norm^2 = 4)
    # For weight-4 vectors: dist^2 = 8 - 2*(u.v) >= 4  =>  u.v <= 2
    # Conflict when u.v > 2, i.e., u.v >= 3
    # This happens when:
    #   - Same support, >= 3 signs agree (dot = 3 or 4)
    #   - Support intersection of size 3 or 4 with enough sign agreement
    
    vectors = []
    supports = list(itertools.combinations(range(D), 4))
    for support in supports:
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    for s4 in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[support[0]] = s1
                        v[support[1]] = s2
                        v[support[2]] = s3
                        v[support[3]] = s4
                        vectors.append(v)
    
    vectors = np.array(vectors)
    N = len(vectors)  # 330 * 16 = 5280
    
    # Precompute dot product matrix
    M = vectors @ vectors.T
    
    # Conflict: M[i,j] > 2 for i != j
    # Build conflict graph as adjacency (sparse representation via bitsets for speed)
    # N=5280, so a dense bool matrix is 5280*5280 = ~27.8M booleans = ~27.8MB. OK.
    conflict = (M > 2)
    np.fill_diagonal(conflict, False)
    
    # For each node, precompute its conflict neighbors as a numpy array
    # Use a list of numpy arrays for fast OR operations
    
    # Strategy: Greedy with multiple orderings + local search (swap-based)
    rng = np.random.default_rng(42)
    
    best_count = 0
    best_indices = []
    
    # Helper: greedy select given an ordering
    def greedy_select(perm, time_limit=None):
        forbidden = np.zeros(N, dtype=bool)
        selected = []
        for idx in perm:
            if forbidden[idx]:
                continue
            selected.append(idx)
            forbidden |= conflict[idx]
            if time_limit and time.time() > time_limit:
                break
        return selected
    
    # Helper: local search via swaps
    def local_search(selected, max_iter=5000, time_limit=None):
        selected_set = set(selected)
        current_count = len(selected)
        
        # Precompute for each selected node, which non-selected nodes it conflicts with
        # and for each non-selected node, which selected nodes it conflicts with
        
        # For swap: remove i from selected, add j to selected
        # Valid if j doesn't conflict with any other selected node (excluding i)
        
        # Precompute conflict sets
        selected_arr = np.array(selected)
        
        # For each candidate j not in selected, count how many selected nodes it conflicts with
        # conflict_count[j] = number of selected nodes i where conflict[i,j]
        
        # Build a count array
        # conflict is N x N boolean
        # For current selected set, conflict_count[j] = sum of conflict[selected, j]
        
        # Efficient: use matrix-vector product
        # sel_indicator: N x 1
        sel_indicator = np.zeros(N, dtype=np.int32)
        sel_indicator[selected_arr] = 1
        
        # conflict_count[j] = conflict @ sel_indicator  (since conflict is symmetric)
        conflict_counts = (conflict @ sel_indicator).astype(np.int32)
        
        # A node j is "addable" if conflict_counts[j] == 0 and j not in selected
        # A swap (remove i, add j) is valid if:
        #   - j not in selected
        #   - conflict_counts[j] <= 1 (it conflicts with at most 1 selected node)
        #   - if conflict_counts[j] == 1, the one it conflicts with must be i
        #   - after removing i, j has 0 conflicts with remaining selected
        
        # Simpler approach: for each pair (i in selected, j not in selected),
        # check if j conflicts only with i among selected nodes.
        
        # Precompute: for each j not in selected, which selected nodes does it conflict with?
        # This is expensive to do naively. Instead:
        
        # conflict_counts[j] tells us how many selected nodes j conflicts with.
        # If conflict_counts[j] == 0, j can be added (but we're doing swaps, not additions).
        # Actually, let's do add/remove separately first, then swaps.
        
        improved = True
        iters = 0
        while improved and iters < max_iter:
            improved = False
            iters += 1
            if time_limit and time.time() > time_limit:
                break
            
            # Recompute conflict counts
            sel_indicator = np.zeros(N, dtype=np.int32)
            sel_arr = np.array(selected)
            if len(sel_arr) > 0:
                sel_indicator[sel_arr] = 1
            conflict_counts = (conflict @ sel_indicator).astype(np.int32)
            
            # Try additions: find j not in selected with conflict_counts[j] == 0
            if len(sel_arr) > 0:
                in_selected = np.zeros(N, dtype=bool)
                in_selected[sel_arr] = True
            else:
                in_selected = np.zeros(N, dtype=bool)
            
            # Candidates for addition
            addable = (conflict_counts == 0) & (~in_selected)
            if np.any(addable):
                # Pick one (first or random)
                add_candidates = np.where(addable)[0]
                j = rng.choice(add_candidates)
                selected.append(int(j))
                improved = True
                continue
            
            # Try swaps: remove i, add j
            # For j not in selected with conflict_counts[j] == 1:
            #   The one selected node it conflicts with is i.
            #   After removing i, j has 0 conflicts. Valid swap.
            # For j not in selected with conflict_counts[j] == 0: already handled above.
            
            # Find nodes with conflict_counts == 1
            swap_candidates_j = (conflict_counts == 1) & (~in_selected)
            if np.any(swap_candidates_j):
                j_candidates = np.where(swap_candidates_j)[0]
                j = int(rng.choice(j_candidates))
                
                # Find which selected node i conflicts with j
                # conflict[i, j] is True for exactly one i in selected
                conflicting_selected = np.where(conflict[j] & in_selected)[0]
                if len(conflicting_selected) == 1:
                    i = int(conflicting_selected[0])
                    # Perform swap
                    selected.remove(i)
                    selected.append(j)
                    improved = True
                    continue
            
            # Try removals that enable additions? (more complex, skip for now)
            # Or try random restarts of local search
            
        return selected
    
    # Phase 1: Multiple greedy runs with different orderings
    num_greedy = 500
    for run in range(num_greedy):
        if time.time() > deadline:
            break
        
        # Different ordering strategies
        if run % 3 == 0:
            perm = rng.permutation(N)
        elif run % 3 == 1:
            # Sort by degree (number of conflicts) ascending
            degrees = conflict.sum(axis=1)
            order = np.argsort(degrees, kind='stable')
            # Add some randomness
            perm = order[rng.permutation(N)]
        else:
            # Random with slight bias toward low-degree nodes
            degrees = conflict.sum(axis=1).astype(float)
            weights = 1.0 / (degrees + 1.0)
            weights /= weights.sum()
            perm = rng.choice(N, size=N, replace=False, p=weights)
        
        selected = greedy_select(perm)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_indices = selected[:]
    
    # Phase 2: Local search on best solution found
    if best_count > 0 and time.time() < deadline:
        improved = local_search(best_indices[:], max_iter=10000, time_limit=deadline)
        if len(improved) > best_count:
            best_count = len(improved)
            best_indices = improved
    
    # Phase 3: More greedy + local search cycles
    for cycle in range(200):
        if time.time() > deadline:
            break
        
        perm = rng.permutation(N)
        selected = greedy_select(perm)
        
        # Quick local search
        if time.time() < deadline - 5:
            selected = local_search(selected, max_iter=2000, time_limit=deadline)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_indices = selected[:]
    
    # Construct the result
    if best_count > 0:
        result = vectors[best_indices]
    else:
        # Fallback to seed solution (weight-2 vectors)
        vectors_seed = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors_seed.append(v)
        result = np.array(vectors_seed, dtype=np.int64)
        
    return result
