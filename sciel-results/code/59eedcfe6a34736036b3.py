import numpy as np
import itertools
import time
import sys

D = 11

def _tomita_max_independent_set(adj_matrix, time_limit):
    """
    Tomita-style branch-and-bound for Maximum Independent Set.
    adj_matrix: numpy boolean matrix where adj[i,j]=True means i and j are adjacent (conflict).
    Returns list of vertex indices forming the maximum independent set.
    """
    n = len(adj_matrix)
    if n == 0:
        return []
    
    # Convert to neighbor sets for faster operations
    neighbors = [np.where(adj_matrix[i])[0] for i in range(n)]
    
    best_size = [0]
    best_set = [[]]
    
    def expand(R, P):
        """
        R: current independent set (list of vertices)
        P: candidate set (list of vertices that can be added to R)
        """
        if time.time() > time_limit:
            return
        
        # Upper bound: |R| + |P|
        if len(R) + len(P) <= best_size[0]:
            return
        
        if len(P) == 0:
            if len(R) > best_size[0]:
                best_size[0] = len(R)
                best_set[0] = list(R)
            return
        
        # Degree-based ordering: sort P by degree within P
        # For efficiency, use a greedy coloring upper bound on the induced subgraph of P
        # But for simplicity and speed, we use |P| as upper bound and rely on ordering
        
        # Order P by degree in P (ascending - Tomita orders by color, but degree is a good proxy)
        # Actually, let's use a simple greedy coloring to get a tighter upper bound
        
        # Compute degrees within P
        p_set = set(P)
        p_degrees = {}
        for v in P:
            deg = 0
            for u in neighbors[v]:
                if u in p_set:
                    deg += 1
            p_degrees[v] = deg
        
        # Sort by degree ascending (process low-degree vertices first for better pruning)
        P_sorted = sorted(P, key=lambda v: p_degrees[v])
        
        for v in P_sorted:
            if time.time() > time_limit:
                return
            
            # Branch: include v
            R.append(v)
            # New P: vertices in P that are not neighbors of v (and not v itself)
            v_neighbors = set(neighbors[v])
            new_P = [u for u in P if u != v and u not in v_neighbors]
            
            expand(R, new_P)
            
            R.pop()
            
            # Remove v from P for the "exclude v" branch
            P.remove(v)
            
            # If P is empty, we're done with this level
            if len(P) == 0:
                break
    
    # Initial call: R=[], P=all vertices
    expand([], list(range(n)))
    
    return best_set[0]

def _greedy_mis(adj_matrix, n, rng):
    """Greedy minimum-degree MIS for initial solution."""
    alive = np.ones(n, dtype=bool)
    selected = []
    degrees = adj_matrix.sum(axis=1).astype(np.int32)
    
    while np.any(alive):
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
        min_deg = np.min(degrees[alive_indices])
        cands = alive_indices[degrees[alive_indices] == min_deg]
        chosen = cands[rng.integers(len(cands))]
        selected.append(chosen)
        neighbors = np.where(adj_matrix[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        alive[to_remove] = False
        rem = np.where(alive)[0]
        if len(rem) > 0:
            degrees[rem] -= adj_matrix[rem][:, to_remove].sum(axis=1)
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
    
    # Compute dot products
    dots = vectors_v4 @ vectors_v4.T
    
    # Conflict if dot > 2
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    rng = np.random.default_rng(42)
    
    # Get initial solution via greedy
    initial_selected = _greedy_mis(conflicts, N4, rng)
    best_selected = initial_selected.copy()
    best_size = len(best_selected)
    
    # Try Tomita branch-and-bound with time limit
    # Use remaining time for B&B
    remaining_time = deadline - time.time() - 5  # 5s safety margin
    
    if remaining_time > 2:
        try:
            # Run Tomita with the remaining time
            # To make it faster, we can use the greedy solution as a lower bound
            # and only explore branches that could beat it
            
            # For now, just run Tomita and see if it improves
            # Note: 5280 vertices might be too large for pure Python Tomita in 125s
            # Let's try with a smaller effective search by using the greedy solution as starting point
            
            # Alternative: use a faster implementation with numpy
            # Let's try a different approach: local search on top of greedy
            
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                selected_arr = np.array(best_selected)
                selected_set = set(best_selected)
                
                # Try to add vertices that don't conflict with selected
                # Find all vertices that don't conflict with any selected
                conflict_with_selected = np.zeros(N4, dtype=bool)
                for idx in selected_arr:
                    conflict_with_selected |= conflicts[idx]
                
                # Candidates: not selected, not conflicting with selected
                candidates = np.where(~conflict_with_selected & ~np.isin(np.arange(N4), selected_arr))[0]
                
                if len(candidates) == 0:
                    break
                
                # Try to find an independent set of size >= 2 from candidates
                # Greedy approach
                cands_list = candidates.tolist()
                rng.shuffle(cands_list)
                
                best_addition = []
                for i, cand1 in enumerate(cands_list):
                    if len(best_addition) >= 2:
                        break
                    # Find another candidate that doesn't conflict with cand1
                    for cand2 in cands_list[i+1:]:
                        if not conflicts[cand1, cand2]:
                            best_addition = [cand1, cand2]
                            break
                    if len(best_addition) >= 2:
                        break
                
                if len(best_addition) >= 2:
                    for cand in best_addition:
                        best_selected.append(cand)
                    improved = True
                    best_size = len(best_selected)
            
            # Try swap-based local search
            if time.time() < deadline - 5:
                improved = True
                while improved and time.time() < deadline - 5:
                    improved = False
                    selected_arr = np.array(best_selected)
                    
                    for i in range(len(selected_arr)):
                        if time.time() > deadline - 5:
                            break
                        remove_idx = selected_arr[i]
                        remaining = np.setdiff1d(selected_arr, [remove_idx])
                        
                        # Find candidates that don't conflict with remaining
                        conflict_with_remaining = np.zeros(N4, dtype=bool)
                        for idx in remaining:
                            conflict_with_remaining |= conflicts[idx]
                        
                        candidates = np.where(~conflict_with_remaining & ~np.isin(np.arange(N4), selected_arr))[0]
                        
                        # Try to add 2+ non-conflicting candidates
                        cands_list = candidates.tolist()
                        rng.shuffle(cands_list)
                        
                        added = []
                        for cand in cands_list:
                            if len(added) >= 2:
                                break
                            ok = True
                            for a in added:
                                if conflicts[cand, a]:
                                    ok = False
                                    break
                            if ok:
                                added.append(cand)
                        
                        if len(added) >= 2:
                            # Perform swap
                            new_selected = list(selected_arr)
                            new_selected.pop(i)
                            new_selected.extend(added)
                            best_selected = new_selected
                            improved = True
                            best_size = len(best_selected)
                            break
            
        except Exception:
            # If B&B or local search fails, keep the greedy solution
            pass
    
    result_v4 = vectors_v4[best_selected]
    
    # Combine V4 and V1
    # V1 dot V4: max dot is 2, so no conflict
    # V1 among themselves: can take both +2 and -2 for each position
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors
