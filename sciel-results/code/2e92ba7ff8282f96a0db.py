import itertools
import numpy as np
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Generate all weight-4 supports
    supports = list(itertools.combinations(range(D), 4))
    n_supports = len(supports)
    
    # Map support to index
    support_to_idx = {s: i for i, s in enumerate(supports)}
    
    # Precompute intersection info
    # inter[i][j] = intersection tuple if len==3 else None
    # To save memory/time, we can just check on the fly or precompute a list of conflicts
    # Conflicts are pairs (i, j, I)
    # But for backtracking, we need to check compatibility of sign s_i with s_j given I.
    
    # Let's precompute for each support i, the list of (j, I) where j < i and |S_i \cap S_j| == 3
    # This allows us to check constraints only against previous variables.
    prev_conflicts = [[] for _ in range(n_supports)]
    for i in range(n_supports):
        Si = supports[i]
        for j in range(i):
            Sj = supports[j]
            inter = tuple(set(Si) & set(Sj))
            if len(inter) == 3:
                prev_conflicts[i].append((j, inter))
                
    # Signs for a support: 16 possibilities
    # Represent sign as a tuple of 4 values in {-1, 1}
    # Or an integer 0..15 mapped to signs
    sign_options = list(itertools.product([-1, 1], repeat=4))
    
    # selected_signs[i] = index of sign in sign_options, or -1 if unassigned
    selected_signs = [-1] * n_supports
    
    # To break symmetry, fix the first support's sign to sign_options[0] (1,1,1,1)
    selected_signs[0] = 0
    
    def get_sign_tuple(sign_idx):
        return sign_options[sign_idx]
        
    def is_compatible(i, sign_idx_i, j, inter):
        """
        Check if sign for support i (index sign_idx_i) is compatible with 
        already selected sign for support j on intersection 'inter'.
        """
        sign_i = get_sign_tuple(sign_idx_i)
        sign_j = get_sign_tuple(selected_signs[j])
        
        Si = supports[i]
        Sj = supports[j]
        
        # We need to compare signs on 'inter'
        # 'inter' is a tuple of indices in the global coordinate system
        # We need to map these to positions in Si and Sj
        # Since inter is subset of Si and Sj, we can find the relative positions.
        
        # Faster: precompute relative positions?
        # For now, just map.
        # Si is a tuple of 4 sorted indices.
        # inter is a tuple of 3 sorted indices.
        
        # Get positions in Si
        pos_in_i = [Si.index(k) for k in inter]
        pos_in_j = [Sj.index(k) for k in inter]
        
        # Compare signs
        for p_i, p_j in zip(pos_in_i, pos_in_j):
            if sign_i[p_i] != sign_j[p_j]:
                return False
        return True

    # Note: is_compatible returns True if signs are DIFFERENT on intersection?
    # Wait. Conflict if signs are SAME.
    # So compatible if signs are DIFFERENT.
    # The code above returns False if ANY sign is different.
    # So it returns True if ALL signs are SAME.
    # So if is_compatible returns True, it's a CONFLICT.
    # We want to avoid conflict. So we need is_compatible to return False.
    
    def has_conflict(i, sign_idx_i):
        for j, inter in prev_conflicts[i]:
            if selected_signs[j] == -1:
                continue
            # Check if signs are SAME on inter
            sign_i = get_sign_tuple(sign_idx_i)
            sign_j = get_sign_tuple(selected_signs[j])
            Si = supports[i]
            Sj = supports[j]
            
            pos_in_i = [Si.index(k) for k in inter]
            pos_in_j = [Sj.index(k) for k in inter]
            
            same = True
            for p_i, p_j in zip(pos_in_i, pos_in_j):
                if sign_i[p_i] != sign_j[p_j]:
                    same = False
                    break
            if same:
                return True
        return False

    def backtrack(idx):
        if time.time() > deadline:
            return False
        
        if idx == n_supports:
            return True
            
        for sign_idx in range(16):
            if not has_conflict(idx, sign_idx):
                selected_signs[idx] = sign_idx
                if backtrack(idx + 1):
                    return True
                selected_signs[idx] = -1
                
        return False

    # Run backtracking
    found = backtrack(1)
    
    if found:
        # Construct vectors
        vectors = []
        for i in range(n_supports):
            sign_idx = selected_signs[i]
            s = get_sign_tuple(sign_idx)
            v = [0] * D
            Si = supports[i]
            for k in range(4):
                v[Si[k]] = s[k]
            vectors.append(v)
        
        # Verify
        vec_arr = np.array(vectors, dtype=np.int64)
        # Check min distance >= max norm
        norms = np.linalg.norm(vec_arr, axis=1)
        max_norm = np.max(norms)
        # Compute pairwise distances
        # 330 vectors, 330x330 distance matrix is small
        dists = np.linalg.norm(vec_arr[:, np.newaxis] - vec_arr[np.newaxis, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist >= max_norm - 1e-9:
            return vec_arr
        else:
            # Should not happen if logic is correct, but fallback to seed
            pass
    else:
        # Fallback to seed solution if backtracking fails (unlikely)
        pass

    # Fallback: Seed solution (weight 2)
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vectors.append(v)
    return np.array(vectors, dtype=np.int64)
