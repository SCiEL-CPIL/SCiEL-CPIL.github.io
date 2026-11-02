import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    # Generate ALL nonzero integer vectors with squared norm <= 4
    candidates = []
    
    # Type 1: One 2, rest 0 (squared norm = 4)
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Two 1s, rest 0 (squared norm = 2)
    for idxs in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    # Type 3: Four 1s, rest 0 (squared norm = 4)
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
    
    # Also add vectors with squared norm = 1 (single 1)
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = s
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # Compute squared norms
    norms_sq = np.sum(candidates ** 2, axis=1)
    
    # Build conflict matrix: 1 if distance_sq < 4
    norms_sq_matrix = norms_sq[:, None] + norms_sq[None, :]
    dist_sq = norms_sq_matrix - 2 * M
    Conflict = (dist_sq < 4).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)
    
    # Precompute neighbor lists
    neighbors = [np.where(Conflict[i] == 1)[0] for i in range(N)]
    degrees = np.array([len(neighbors[i]) for i in range(N)])
    
    best_set = []
    best_size = 0
    
    def run_greedy_and_local(seed):
        if time.time() > deadline:
            return []
            
        rng = np.random.default_rng(seed)
        
        # Greedy initialization
        in_selected = np.zeros(N, dtype=bool)
        
        # Order by degree, then random tie-break
        order = np.argsort(degrees + rng.random(N))
        
        for i in order:
            # Check if i conflicts with any already selected
            # We can check neighbors
            conflict_found = False
            for n in neighbors[i]:
                if in_selected[n]:
                    conflict_found = True
                    break
            
            if not conflict_found:
                in_selected[i] = True
        
        # Local Search
        # Initialize conflict_count
        conflict_count = np.zeros(N, dtype=np.int32)
        for i in range(N):
            if in_selected[i]:
                conflict_count[neighbors[i]] += 1
        conflict_count[in_selected] = 0
        
        improved = True
        max_iters = 300
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            unselected = np.where(~in_selected)[0]
            if len(unselected) == 0:
                break
                
            # 1. Try to add a vertex with 0 conflicts
            valid_adds = unselected[conflict_count[unselected] == 0]
            if len(valid_adds) > 0:
                v = rng.choice(valid_adds)
                in_selected[v] = True
                conflict_count[neighbors[v]] += 1
                conflict_count[v] = 0
                improved = True
                continue # Restart loop
            
            # 2. Try a swap: remove u, add v
            # Look for v with conflict_count 1
            valid_swaps = unselected[conflict_count[unselected] == 1]
            if len(valid_swaps) > 0:
                v = rng.choice(valid_swaps)
                # Find unique neighbor u in selected
                u = -1
                for n in neighbors[v]:
                    if in_selected[n]:
                        u = n
                        break
                
                if u != -1:
                    # Swap
                    in_selected[u] = False
                    conflict_count[neighbors[u]] -= 1
                    conflict_count[u] = 0
                    
                    in_selected[v] = True
                    conflict_count[neighbors[v]] += 1
                    conflict_count[v] = 0
                    improved = True
                    continue # Restart loop
            
            # 3. Perturbation if stuck
            if iters > 50:
                # Remove 2 random selected vertices
                selected_idx = np.where(in_selected)[0]
                if len(selected_idx) > 0:
                    remove_idx = rng.choice(len(selected_idx), size=min(2, len(selected_idx)), replace=False)
                    for idx in remove_idx:
                        u = selected_idx[idx]
                        in_selected[u] = False
                        conflict_count[neighbors[u]] -= 1
                        conflict_count[u] = 0
                    improved = True
                    continue
                    
        return np.where(in_selected)[0]

    # Run multiple restarts
    num_restarts = 200
    for s in range(num_restarts):
        if time.time() > deadline:
            break
        
        res = run_greedy_and_local(s)
        
        if len(res) > best_size:
            best_size = len(res)
            best_set = res

    if not best_set:
        best_set = list(range(min(22, N)))
        
    return candidates[best_set].astype(np.int64)
