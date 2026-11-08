import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Generate W3 vectors
    # Support: combinations of 3 indices
    # Signs: 8 combinations
    supports = list(itertools.combinations(range(D), 3))
    V3_list = []
    for supp in supports:
        for signs in itertools.product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(supp, signs):
                v[idx] = s
            V3_list.append(v)
    V3 = np.array(V3_list, dtype=np.int64)
    n3 = len(V3)
    
    # Compute dot products for V3
    # D33 = V3 @ V3.T
    # Conflict if D33 > 1
    D33 = V3 @ V3.T
    conflict_mask = D33 > 1
    np.fill_diagonal(conflict_mask, False)
    
    best_set = None
    best_size = 0
    
    # Greedy MIS
    # We can optimize by pre-computing neighbor lists?
    # But boolean array operations are fast.
    
    for _ in range(5000):
        if time.time() > deadline:
            break
            
        selected = np.zeros(n3, dtype=bool)
        available = np.ones(n3, dtype=bool)
        order = np.random.permutation(n3)
        
        for i in order:
            if not available[i]:
                continue
            # Check conflicts with selected
            # conflict_mask[i] is a boolean vector
            # We need to check if any selected node is in conflict_mask[i]
            # np.any(conflict_mask[i] & selected)
            if np.any(conflict_mask[i] & selected):
                continue
            
            selected[i] = True
            # Remove conflicts from available
            available[conflict_mask[i]] = False
            available[i] = False
            
        size = np.sum(selected)
        if size > best_size:
            best_size = size
            best_set = selected.copy()
            
    if best_set is None:
        # Fallback to seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    V_sel = V3[best_set]
    
    # Try to add W1 vectors
    # W1: e_i and -e_i
    V1_list = []
    for i in range(D):
        v_pos = np.zeros(D, dtype=np.int64)
        v_pos[i] = 1
        V1_list.append(v_pos)
        
        v_neg = np.zeros(D, dtype=np.int64)
        v_neg[i] = -1
        V1_list.append(v_neg)
    V1 = np.array(V1_list, dtype=np.int64)
    
    # Check compatibility with V_sel
    # Condition: v1 . v3 <= 0
    # D13 = V1 @ V_sel.T
    # If max(D13[j, :]) <= 0, then V1[j] is compatible.
    if len(V_sel) > 0:
        D13 = V1 @ V_sel.T
        compat_1 = np.all(D13 <= 0, axis=1)
        V1_add = V1[compat_1]
    else:
        V1_add = V1
        
    # Try to add W2 vectors
    # Condition: v2 . v3 <= 1
    V2_list = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                V2_list.append(v)
    V2 = np.array(V2_list, dtype=np.int64)
    
    # Check compatibility with V_sel and V1_add
    # First with V_sel
    if len(V_sel) > 0:
        D23 = V2 @ V_sel.T
        compat_2_3 = np.all(D23 <= 1, axis=1)
    else:
        compat_2_3 = np.ones(len(V2), dtype=bool)
        
    V2_cand = V2[compat_2_3]
    
    # Check compatibility with V1_add
    # Condition: v2 . v1 <= 0
    if len(V1_add) > 0 and len(V2_cand) > 0:
        D21 = V2_cand @ V1_add.T
        # We need to select a subset of V2_cand that is compatible with V1_add
        # And mutually compatible (which they are).
        # So for each v2 in V2_cand, check if v2 . v1 <= 0 for all v1 in V1_add.
        compat_2_1 = np.all(D21 <= 0, axis=1)
        V2_add = V2_cand[compat_2_1]
    else:
        V2_add = V2_cand
        
    # Combine
    V_final_list = [V_sel, V1_add, V2_add]
    V_final = np.vstack([v for v in V_final_list if len(v) > 0])
    
    # Verify
    # Check min distance >= max norm
    if len(V_final) > 0:
        norms_sq = np.sum(V_final**2, axis=1)
        max_norm_sq = np.max(norms_sq)
        
        # Check pairwise distances
        # dist_sq = |u|^2 + |v|^2 - 2 u.v
        # We need dist_sq >= max_norm_sq
        # This is O(N^2). N might be ~200-300. 90k pairs. Fast.
        D_dot = V_final @ V_final.T
        # dist_sq[i, j] = norms_sq[i] + norms_sq[j] - 2 * D_dot[i, j]
        dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * D_dot
        np.fill_diagonal(dist_sq, 0)
        
        min_dist_sq = np.min(dist_sq)
        
        if min_dist_sq < max_norm_sq:
            # Invalid, fallback to seed
            vectors = []
            for i, j in itertools.combinations(range(D), 2):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = [0] * D
                        v[i], v[j] = si, sj
                        vectors.append(v)
            return np.array(vectors, dtype=np.int64)
            
    return V_final
