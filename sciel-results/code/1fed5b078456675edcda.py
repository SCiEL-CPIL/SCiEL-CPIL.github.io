"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _is_compatible(v, selected_vecs, selected_norms):
    """
    Check if vector v is compatible with all vectors in selected_vecs.
    Compatibility condition: ||v - w||^2 >= max(||v||^2, ||w||^2)
    which simplifies to:
    ||v||^2 + ||w||^2 - 2 v.w >= max(||v||^2, ||w||^2)
    => 2 v.w <= min(||v||^2, ||w||^2)
    """
    n_v = np.dot(v, v)
    # Vectorized check against all selected vectors
    # selected_vecs is (k, D), v is (D,)
    dots = selected_vecs @ v
    # Condition: 2 * dots <= min(n_v, selected_norms)
    # min(n_v, selected_norms) is an array of shape (k,)
    limits = np.minimum(n_v, selected_norms)
    return np.all(2 * dots <= limits)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit
    
    # 1. Generate a diverse pool of candidate vectors
    # We use integer vectors with small norms.
    # Norms 1, 2, 3, 4 are good candidates.
    # Norm 1: 22 vectors (unit vectors +/-)
    # Norm 2: 2*11*10 = 220 vectors (two non-zero entries)
    # Norm 3: 11*10*9*2 = 1980 vectors (three non-zero entries, all 1/-1)
    # Norm 4: 
    #   Case 1: four 1/-1 entries: 11*10*9*8 * 2^4 / 24? No, combinations with signs.
    #            C(11,4) * 2^4 = 330 * 16 = 5280
    #   Case 2: one 2, rest 0: 11 * 2 = 22
    #   Case 3: two 2s, rest 0: C(11,2) * 2^2 = 55 * 4 = 220
    #   Case 4: one 1, one sqrt(3)? No, integer vectors.
    #   Case 5: one 2, two 1s? 4+1+1=6 > 4.
    
    candidates = []
    
    # Norm 1
    for i in range(D):
        v1 = np.zeros(D, dtype=np.int64); v1[i] = 1; candidates.append(v1)
        v2 = np.zeros(D, dtype=np.int64); v2[i] = -1; candidates.append(v2)
        
    # Norm 2
    for i in range(D):
        for j in range(i+1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si; v[j] = sj
                    candidates.append(v)
                    
    # Norm 3
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for si in [1, -1]:
                    for sj in [1, -1]:
                        for sk in [1, -1]:
                            v = np.zeros(D, dtype=np.int64)
                            v[i] = si; v[j] = sj; v[k] = sk
                            candidates.append(v)
                            
    # Norm 4 (Type 1: four 1/-1)
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for l in range(k+1, D):
                    for si in [1, -1]:
                        for sj in [1, -1]:
                            for sk in [1, -1]:
                                for sl in [1, -1]:
                                    v = np.zeros(D, dtype=np.int64)
                                    v[i] = si; v[j] = sj; v[k] = sk; v[l] = sl
                                    candidates.append(v)
                                    
    # Norm 4 (Type 2: one 2)
    for i in range(D):
        for si in [1, -1]:
            v = np.zeros(D, dtype=np.int64)
            v[i] = si * 2
            candidates.append(v)
            
    # Norm 4 (Type 3: two 2s)
    for i in range(D):
        for j in range(i+1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si * 2; v[j] = sj * 2
                    candidates.append(v)

    candidates = np.array(candidates)
    n_cand = len(candidates)
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # Precompute pairwise dot products? 
    # n_cand is roughly 22 + 220 + 1980 + 5280 + 22 + 220 = 7744.
    # Dot matrix 7744x7744 is ~60MB. Feasible.
    # But we need dynamic selection. 
    # Let's stick to on-the-fly dot products for the selected set, 
    # but pre-compute a "conflict graph" is too big for memory/time if we want to be safe.
    # However, checking compatibility of a new vector against K selected vectors is O(K*D).
    # If K is ~500, D=11, that's 5500 ops. Very fast.
    
    # 2. Initialize with a greedy solution
    # Use random restarts to find a good initial set for SA
    best_set = []
    best_norms = []
    best_count = 0
    rng = np.random.default_rng(42)
    
    # Quick greedy initialization
    for restart in range(20):
        if time.time() > deadline - 120: break
        perm = rng.permutation(n_cand)
        curr_set = []
        curr_norms = []
        for idx in perm:
            v = candidates[idx]
            if _is_compatible(v, np.array(curr_set) if curr_set else np.empty((0,D)), np.array(curr_norms) if curr_norms else np.empty(0)):
                curr_set.append(v)
                curr_norms.append(norms_sq[idx])
        if len(curr_set) > best_count:
            best_count = len(curr_set)
            best_set = curr_set.copy()
            best_norms = curr_norms.copy()
            
    # 3. Simulated Annealing
    # State: set of selected vectors
    # Move: 
    #   1. Try to insert a random candidate. If compatible, accept.
    #   2. If not compatible, identify a conflicting vector from the current set.
    #      Remove the conflicting vector and insert the new one (if the new one is "better" or by probability).
    #      Actually, simpler: 
    #      - Pick a random candidate v_new.
    #      - Check compatibility.
    #      - If compatible: Add it. (Always accept)
    #      - If not: 
    #          Find a random conflicting vector v_old in the set.
    #          Remove v_old.
    #          Check if v_new is compatible with the remaining set.
    #          If yes: Add v_new. (Swap)
    #          If no: Reject (or rollback).
    #          
    # To improve, we can also try removing a random vector and not adding anything (unlikely to help size).
    # Or adding a vector that requires removing MULTIPLE vectors? Too complex.
    # Let's stick to Insert and Swap.
    
    current_set = best_set
    current_norms = best_norms
    current_count = len(current_set)
    
    # Convert to lists of indices for faster manipulation? 
    # We need the vectors themselves for dot products.
    # Let's keep arrays.
    
    # We need a way to quickly check compatibility.
    # If we keep current_set as a list of numpy arrays, appending/removing is O(1) amortized.
    # Checking compatibility is O(K*D).
    
    # To speed up SA, we can pre-compute a "compatibility list" for each candidate?
    # No, the set changes.
    
    # Optimization: 
    # Instead of checking against ALL K vectors, we can maintain a count of "conflicts" or just check.
    # With K~500 and D=11, a check is ~5500 flops. 
    # In Python, a loop of 500 dot products might be slow if done naively.
    # np.dot is fast.
    
    # Let's optimize the compatibility check by keeping the selected vectors in a numpy array 
    # and updating it. But inserting/deleting in the middle of a numpy array is O(K).
    # Since K is small (<1000), O(K) shift is fine.
    
    # Better: Keep a list of indices into `candidates`.
    # And a list of vectors.
    
    # Let's re-initialize current_set as list of indices to save memory and copying.
    curr_indices = []
    for v in best_set:
        # Find index of v in candidates? Expensive.
        # Let's just use the vectors directly.
        pass
    
    # Let's restart the SA state from the best greedy result found above.
    # We have best_set (list of np.arrays) and best_norms (list of floats).
    
    T = 1.0
    T_min = 1e-6
    alpha = 0.999 # Cooling rate
    
    # We need to be able to pick a random conflicting vector.
    # So we need to know WHICH vectors conflict with v_new.
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Pick a random candidate to try to insert
        idx_new = rng.integers(0, n_cand)
        v_new = candidates[idx_new]
        n_new = norms_sq[idx_new]
        
        if not current_set:
            current_set.append(v_new)
            current_norms.append(n_new)
            continue
            
        sel_arr = np.array(current_set)
        sel_norms_arr = np.array(current_norms)
        
        dots = sel_arr @ v_new
        limits = np.minimum(n_new, sel_norms_arr)
        conflicts = 2 * dots > limits
        
        if not np.any(conflicts):
            # Compatible! Add it
            current_set.append(v_new)
            current_norms.append(n_new)
            if len(current_set) > best_count:
                best_count = len(current_set)
                best_set = current_set.copy()
                best_norms = current_norms.copy()
        else:
            # Conflicts exist
            # Try a swap: remove a conflicting vector, add v_new
            # Pick a random conflicting index
            conflict_indices = np.where(conflicts)[0]
            idx_conf = conflict_indices[rng.integers(0, len(conflict_indices))]
            
            # Remove conflicting vector
            v_old = current_set[idx_conf]
            n_old = current_norms[idx_conf]
            
            # Temporarily remove
            current_set.pop(idx_conf)
            current_norms.pop(idx_conf)
            
            # Check if v_new is now compatible with the rest
            if current_set:
                sel_arr_rest = np.array(current_set)
                sel_norms_rest = np.array(current_norms)
                dots_rest = sel_arr_rest @ v_new
                limits_rest = np.minimum(n_new, sel_norms_rest)
                if np.all(2 * dots_rest <= limits_rest):
                    # Accept swap
                    current_set.append(v_new)
                    current_norms.append(n_new)
                else:
                    # Reject swap, restore v_old
                    current_set.append(v_old)
                    current_norms.append(n_old)
            else:
                # Set was empty (size 1 before removal)
                current_set.append(v_new)
                current_norms.append(n_new)
                
        # Update best
        if len(current_set) > best_count:
            best_count = len(current_set)
            best_set = current_set.copy()
            best_norms = current_norms.copy()
            
        # Cool down
        T *= alpha
        
    return np.array(best_set, dtype=np.int64)

# EVOLVE-BLOCK-END
