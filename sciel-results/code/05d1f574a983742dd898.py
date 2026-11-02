import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    candidates = []
    
    # Norm 4
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    # Norm 8
    for idxs in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = 2 * s
            candidates.append(v)
            
    for i in range(D):
        for si in [1, -1]:
            for idxs in itertools.combinations([j for j in range(D) if j != i], 2):
                for signs in itertools.product([1, -1], repeat=2):
                    v = [0] * D
                    v[i] = 2 * si
                    for idx, s in zip(idxs, signs):
                        v[idx] = s
                    candidates.append(v)
                    
    for idxs in itertools.combinations(range(D), 8):
        for signs in itertools.product([1, -1], repeat=8):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    norms_sq = np.sum(candidates**2, axis=1)
    
    best_set_indices = []
    rng = np.random.default_rng(42)
    
    # Use a smaller number of restarts if N is large
    num_restarts = 20 if N > 10000 else 100
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        order = rng.permutation(N)
        selected = []
        selected_vectors = [] # List of numpy arrays
        selected_norms = []
        
        # To speed up, we can break early if we can't beat current best?
        # Or just run.
        
        for idx in order:
            if not selected:
                selected.append(idx)
                selected_vectors.append(candidates[idx])
                selected_norms.append(norms_sq[idx])
                continue
                
            # Check compatibility
            # Convert selected_vectors to array for dot product
            # This is inefficient if done every time.
            # Better: maintain a numpy array of selected vectors.
            
            # Let's re-structure:
            # selected_indices = []
            # selected_vecs = np.array([])
            
            # Inside loop:
            # if len(selected_vecs) == 0: ...
            # else:
            #   dots = selected_vecs @ candidates[idx]
            #   limits = ...
            
            # This requires updating selected_vecs which is expensive (concatenation).
            # Alternative: keep a list, and convert to array only for the check?
            # Or pre-allocate a large array and keep a count?
            
            pass # Placeholder for logic
            
    # The above loop is too slow in Python for N=50000.
    
    # We must use the matrix method for a subset or a very efficient C-extension like NumPy.
    # Given the constraints, the subset approach (Norm 4 + Two 2s) is the safest for speed.
    # Let's stick to the subset that fits in memory for the matrix.
    
    # Subset: Norm 4 (5302) + Two 2s (220) = 5522.
    # Matrix 5522x5522 is 120MB.
    
    # If the subset result is < 220, we return it.
    # If it is > 220, we return it.
    
    # Let's implement the subset search properly.
    
    mask_subset = (norms_sq == 4) | ((norms_sq == 8) & (np.sum(candidates**2 == 4, axis=1) == 2))
    subset_indices = np.where(mask_subset)[0]
    subset = candidates[subset_indices]
    N_sub = len(subset)
    norms_sub = norms_sq[subset_indices]
    
    M_sub = subset @ subset.T
    
    best_set_sub = []
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        order = rng.permutation(N_sub)
        selected = []
        selected_mask = np.zeros(N_sub, dtype=bool)
        
        for idx in order:
            if not selected:
                selected.append(idx)
                selected_mask[idx] = True
                continue
                
            dots = M_sub[idx, selected_mask]
            
            # Limits
            # If both are norm 8, limit is 4. Else 2.
            # We need to know which selected are norm 8.
            # Precompute is_norm8 for subset
            is_norm8 = (norms_sub == 8)
            
            # For current idx:
            norm_idx = norms_sub[idx]
            is_norm8_idx = (norm_idx == 8)
            
            # Limit for each selected j:
            # limit_j = 4 if (is_norm8[j] and is_norm8_idx) else 2
            
            # We can compute this vector of limits
            # limits = np.where(is_norm8[sel_indices] & is_norm8_idx, 4, 2)
            
            # But we need sel_indices.
            # We can maintain a list of selected indices to get norms easily.
            
            # Optimization:
            # If is_norm8_idx is False, all limits are 2.
            # Check: np.all(dots <= 2)
            
            # If is_norm8_idx is True:
            # Check: np.all(dots <= 4) AND for those with dot > 2, they must be norm 8.
            # i.e., np.all(dots <= 4) AND np.all((dots > 2) | (~is_norm8[sel_indices]))
            # Or: np.all(dots <= np.where(is_norm8[sel_indices], 4, 2))
            
            if not is_norm8_idx:
                if np.all(dots <= 2):
                    selected.append(idx)
                    selected_mask[idx] = True
            else:
                sel_indices = np.where(selected_mask)[0]
                limits = np.where(is_norm8[sel_indices], 4, 2)
                if np.all(dots <= limits):
                    selected.append(idx)
                    selected_mask[idx] = True
                    
        if len(selected) > len(best_set_sub):
            best_set_sub = selected
            
    if not best_set_sub:
        return candidates[:22].astype(np.int64)
        
    # Map back to original indices
    final_indices = subset_indices[best_set_sub]
    return candidates[final_indices].astype(np.int64)
