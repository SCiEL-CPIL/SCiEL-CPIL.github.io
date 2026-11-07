import numpy as np
import time
import random

def construct():
    n = 12
    m = 40
    total_subsets = 1 << n
    deadline = time.time() + 85.0
    
    # Precompute popcount
    sizes = np.array([bin(i).count('1') for i in range(total_subsets)], dtype=np.int32)
    
    # Build comparability matrix C
    # C[i, j] = 1 if i and j are comparable (i strict subset j OR j strict subset i)
    C = np.zeros((total_subsets, total_subsets), dtype=np.int8)
    
    # Efficient construction of C
    # i is subset of j iff (i & j) == i
    # We can iterate i and update all j supersets of i
    # But vectorized boolean ops on chunks are faster in numpy
    
    chunk_size = 1024
    for start in range(0, total_subsets, chunk_size):
        end = min(start + chunk_size, total_subsets)
        i_chunk = np.arange(start, end, dtype=np.int32)
        j_all = np.arange(total_subsets, dtype=np.int32)
        
        # i_chunk[:, None] vs j_all[None, :]
        # This creates a 1024 x 4096 matrix of bools. ~4M bools. OK.
        i_exp = i_chunk[:, None]
        j_exp = j_all[None, :]
        
        # i is strict subset of j
        is_sub = (i_exp & j_exp) == i_exp
        not_eq = i_exp != j_exp
        M_ij = (is_sub & not_eq).astype(np.int8)
        
        # j is strict subset of i
        # (j & i) == j and j != i
        is_sub_rev = (j_exp & i_exp) == j_exp
        not_eq_rev = j_exp != i_exp
        M_ji = (is_sub_rev & not_eq_rev).astype(np.int8)
        
        C[start:end, :] = (M_ij | M_ji).astype(np.int8)
        
    # Symmetric check (C should be symmetric)
    # C = C | C.T (just in case of numerical issues, though logic is exact)
    
    def compute_score(family):
        if len(family) < 2:
            return 0
        fam_arr = np.array(family, dtype=np.int32)
        # Get upper triangle indices
        idx_i, idx_j = np.triu_indices(len(fam_arr), k=1)
        pairs = C[fam_arr[idx_i], fam_arr[idx_j]]
        return int(np.sum(pairs))
    
    def local_search(family, deadline, max_iters=1000):
        family = list(family)
        used = set(family)
        current_score = compute_score(family)
        
        # Precompute list of all masks not in family for quick access
        # We will update this list as we swap
        
        for it in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            
            best_gain = 0
            best_swap = None
            best_new_score = current_score
            
            # Try swapping each element in family with each element NOT in family
            # This is O(N^2 * 40) which is expensive.
            # Optimization: Calculate gain for removing x and adding y.
            # Gain = (comp(y, family \ {x})) - (comp(x, family \ {x}))
            
            fam_arr = np.array(family, dtype=np.int32)
            n_fam = len(family)
            
            # comp_count[x] for x in family: number of other family members comparable with x
            # This is the row sum of C[fam, fam] minus 1 (self)
            # But we need it for each x.
            # Let's compute the contribution of each x to the total score.
            # contrib[x] = sum_{y in family, y!=x} C[x,y]
            
            # We can compute this vectorized
            # M_fam = C[np.ix_(fam_arr, fam_arr)]
            # contrib = M_fam.sum(axis=1) - 1 (diagonal is 0 anyway, so sum is fine)
            
            M_fam = C[np.ix_(fam_arr, fam_arr)]
            contribs = M_fam.sum(axis=1)
            
            # For each x in family, score_without_x = current_score - contribs[x]
            
            # For each y NOT in family, we want max_y (score_without_x + comp(y, family \ {x}))
            # comp(y, family \ {x}) = sum_{z in family, z!=x} C[y,z]
            
            # Let's iterate x
            # To speed up, we can just do a few random swaps or the best swap found in a subset
            # Given time constraints, we'll do a limited search
            
            # Sample a few x's and y's?
            # Or just try all x, and for each x, find best y.
            # Finding best y for a fixed x:
            # We need C[y, family] sum, minus C[y, x] (since x is removed)
            # Let S_y = C[y, fam_arr].sum()
            # Then comp(y, family \ {x}) = S_y - C[y, x]
            
            # Precompute S_y for all y not in family?
            # This is a 4096 x 40 matrix.
            # C[all_masks, fam_arr].sum(axis=1)
            
            all_masks = np.arange(total_subsets, dtype=np.int32)
            # S_y for all y
            # This is expensive to recompute every iteration.
            # But we only swap one element at a time.
            
            # Let's just do a simple hill climb:
            # Pick a random x from family.
            # Find best y not in family to replace it.
            
            x_idx = random.randint(0, n_fam - 1)
            x = family[x_idx]
            
            score_without_x = current_score - int(contribs[x_idx])
            
            # Find best y
            # We need C[y, family \ {x}]
            # Let's compute C[all_masks, fam_arr]
            # This is 4096 x 40.
            C_all_fam = C[all_masks, fam_arr]
            sums = C_all_fam.sum(axis=1)
            
            # For y=x, we don't care.
            # For y != x, comp(y, family \ {x}) = sums[y] - C[y, x]
            # C[y, x] is the column x of C[all_masks, x]
            C_y_x = C[all_masks, x]
            
            potential_gains = sums - C_y_x
            
            # We cannot pick y if y is in family (except x, which is being removed)
            # Mask out family members
            mask = np.ones(total_subsets, dtype=bool)
            mask[fam_arr] = False
            mask[x] = True # x is allowed to be "picked" effectively, but it's the same element.
            # Actually, we want y NOT in family.
            # So mask out all family members.
            mask[fam_arr] = False
            
            potential_gains[~mask] = -10000
            
            best_y_idx = np.argmax(potential_gains)
            best_y = int(all_masks[best_y_idx])
            gain = int(potential_gains[best_y_idx])
            
            new_score = score_without_x + gain
            
            if new_score > current_score:
                # Perform swap
                family[x_idx] = best_y
                used.discard(x)
                used.add(best_y)
                current_score = new_score
            else:
                # No improvement with this random x
                # Try a few more random x's?
                # To save time, we break if no improvement after a few tries
                # But we only tried 1 x.
                # Let's try 5 random x's per iteration
                improved = False
                for _ in range(4):
                    x_idx = random.randint(0, n_fam - 1)
                    x = family[x_idx]
                    score_without_x = current_score - int(contribs[x_idx])
                    
                    # Reuse C_all_fam? It's still valid for the current family.
                    # But contribs changed? No, family didn't change.
                    # C_y_x changes with x.
                    C_y_x = C[all_masks, x]
                    potential_gains = sums - C_y_x
                    potential_gains[~mask] = -10000
                    best_y_idx = np.argmax(potential_gains)
                    best_y = int(all_masks[best_y_idx])
                    gain = int(potential_gains[best_y_idx])
                    new_score = score_without_x + gain
                    if new_score > current_score:
                        family[x_idx] = best_y
                        used.discard(x)
                        used.add(best_y)
                        current_score = new_score
                        improved = True
                        break
                
                if not improved:
                    # Local optimum reached
                    break
        
        return family, current_score
    
    # Initializations
    inits = []
    
    # Init 1: 3 chains of 13
    # Chain 1: 0, 1, 3, 7...
    c1 = [(1 << i) - 1 for i in range(13)]
    # Chain 2: 2, 6, 14... (shifted by 1 bit, but must be distinct)
    # {1}, {1,2}, {1,2,3}...
    c2 = []
    cur = 0
    for i in range(1, 13):
        cur |= (1 << i)
        c2.append(cur)
    # Check distinctness with c1
    # c1 has {0}, {0,1}...
    # c2 has {1}, {1,2}...
    # Intersection: None?
    # {1} not in c1. {1,2} not in c1.
    # So c1 and c2 are disjoint.
    
    # Chain 3: {2}, {2,3}...
    c3 = []
    cur = 0
    for i in range(2, 13):
        cur |= (1 << i)
        c3.append(cur)
    
    # We have 13 + 12 + 11 = 36 sets.
    # Need 4 more.
    # Add {3}, {3,4}, {3,4,5}, {3,4,5,6}
    c4 = []
    cur = 0
    for i in range(3, 7):
        cur |= (1 << i)
        c4.append(cur)
        
    init1 = c1 + c2 + c3 + c4
    init1 = list(dict.fromkeys(init1))
    while len(init1) < 40:
        init1.append(random.randint(0, total_subsets-1))
        init1 = list(dict.fromkeys(init1))
    inits.append(init1[:40])
    
    # Init 2: Random
    inits.append(random.sample(range(total_subsets), 40))
    
    # Init 3: Greedy from empty
    init3 = [0, (1<<12)-1]
    for i in range(1, 12):
        init3.append(1 << i)
    while len(init3) < 40:
        init3.append(random.randint(0, total_subsets-1))
        init3 = list(dict.fromkeys(init3))
    inits.append(init3[:40])
    
    best_score = -1
    best_family = None
    
    for init in inits:
        if time.time() > deadline - 10.0:
            break
        fam, score = local_search(init, deadline, max_iters=200)
        if score > best_score:
            best_score = score
            best_family = fam
    
    # Final polish
    if best_family and time.time() < deadline - 5.0:
        best_family, best_score = local_search(best_family, deadline, max_iters=500)
        
    if best_family is None:
        best_family = list(range(40))
        
    result = []
    for mask in best_family:
        mask = int(mask)
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
        
    return result
