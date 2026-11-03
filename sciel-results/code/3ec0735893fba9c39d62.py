import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    D = 11
    
    # Type A: Vectors with one non-zero entry +/- 2
    # Norm squared = 4
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)
    
    # Type D: Vectors with four non-zero entries +/- 1
    # Norm squared = 4
    vectors_D = []
    indices = list(range(D))
    for combo in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(combo, signs):
                v[idx] = sign
            vectors_D.append(v)
            
    vectors_D = np.array(vectors_D, dtype=np.int8)
    
    # Compute conflict matrix for Type D
    # Conflict if dot product >= 3
    # Dot product range: -4 to 4
    # We need distance >= 2 => dist^2 >= 4
    # dist^2 = 4 + 4 - 2 dot = 8 - 2 dot
    # 8 - 2 dot >= 4 => 2 dot <= 4 => dot <= 2
    # So conflict if dot > 2, i.e., dot >= 3
    
    # Optimization: Only compute upper triangle or use vectorized ops
    # 5280 x 5280 dot product
    # This might be memory heavy if not careful, but int8 -> int16/32 is fine.
    # 5280*5280*4 bytes ~ 110MB. Acceptable.
    
    dot_products = vectors_D @ vectors_D.T
    
    # Mask for conflicts (dot >= 3)
    # Note: Diagonal is 4, which is a conflict with itself, but we handle selection logic.
    conflict_mask = dot_products >= 3
    
    N_D = len(vectors_D)
    
    # Precompute neighbor lists for speed
    # neighbor_list[i] = set of j such that conflict_mask[i, j]
    neighbor_lists = []
    for i in range(N_D):
        conflicts = np.where(conflict_mask[i])[0]
        neighbor_lists.append(conflicts)
    
    # Also precompute degrees for degree-based greedy
    degrees = np.array([len(nl) for nl in neighbor_lists])
    
    best_set_D = []
    
    num_restarts = 200
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        available = set(range(N_D))
        current_set = []
        
        while available:
            if time.time() > deadline:
                break
                
            # Strategy: pick vertex with minimum degree among available
            # For speed, sample a subset if available is large
            if len(available) > 200:
                # Sample 50 random candidates and pick min degree
                candidates = list(np.random.choice(list(available), size=50, replace=False))
                min_deg = min(degrees[c] for c in candidates)
                # Among those with min degree, pick one randomly
                min_cands = [c for c in candidates if degrees[c] == min_deg]
                v_idx = min_cands[np.random.randint(len(min_cands))]
            else:
                min_deg = min(degrees[c] for c in available)
                min_cands = [c for c in available if degrees[c] == min_deg]
                v_idx = min_cands[np.random.randint(len(min_cands))]
            
            current_set.append(v_idx)
            available.remove(v_idx)
            
            # Remove all neighbors from available
            available.difference_update(neighbor_lists[v_idx])
            
        if len(current_set) > len(best_set_D):
            best_set_D = current_set
            
    # Local improvement: try to add back vertices not in best_set_D
    if best_set_D:
        selected_set = set(best_set_D)
        # Try to add vertices that don't conflict with any selected vertex
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for v in range(N_D):
                if v in selected_set:
                    continue
                # Check if v conflicts with any selected vertex
                conflicts_with_selected = False
                for nb in neighbor_lists[v]:
                    if nb in selected_set:
                        conflicts_with_selected = True
                        break
                if not conflicts_with_selected:
                    selected_set.add(v)
                    improved = True
        best_set_D = list(selected_set)
            
    # Combine Type A and selected Type D
    vectors_A = np.array(vectors_A, dtype=np.int64)
    if best_set_D:
        vectors_D_selected = vectors_D[best_set_D].astype(np.int64)
        final_vectors = np.vstack([vectors_A, vectors_D_selected])
    else:
        final_vectors = vectors_A
    
    return final_vectors
