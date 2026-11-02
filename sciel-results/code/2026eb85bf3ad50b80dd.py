import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    D = 11
    
    # Generate all integer vectors with norm squared 4
    # Type 1: 2 * e_i
    # Type 2: +/- e_i +/- e_j +/- e_k +/- e_l
    
    vectors = []
    
    # Type 1: 2 * e_i
    for i in range(D):
        v = np.zeros(D, dtype=np.int64)
        v[i] = 2
        vectors.append(v)
        v[i] = -2
        vectors.append(v)
        
    # Type 2: four 1s
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([-1, 1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(support, signs):
                v[pos] = sign
            vectors.append(v)
            
    vecs = np.array(vectors, dtype=np.float64)
    n_vecs = len(vecs)
    
    # Precompute all pairwise dot products for fast compatibility checks
    # n_vecs ~ 5300, so 5300^2 / 2 ~ 14M entries - too large for memory
    # Instead, we'll compute on-the-fly but use vectorized operations
    
    best_set = []
    best_count = 0
    
    # Phase 1: Greedy to get initial solutions
    max_greedy_restarts = 20
    initial_solutions = []
    
    for r in range(max_greedy_restarts):
        if time.time() > deadline - 40:
            break
            
        rng = np.random.RandomState(r * 1000 + 42)
        indices = rng.permutation(n_vecs)
        
        selected_indices = []
        selected_vectors = []
        
        for idx in indices:
            if time.time() > deadline - 35:
                break
                
            v = vecs[idx]
            
            if len(selected_vectors) > 0:
                # Vectorized compatibility check
                sel_arr = np.array(selected_vectors)
                dots = sel_arr @ v
                if np.any(dots > 2):
                    continue
            
            selected_indices.append(idx)
            selected_vectors.append(v)
        
        if len(selected_indices) > best_count:
            best_count = len(selected_indices)
            best_set = selected_indices[:]
        
        initial_solutions.append(selected_indices[:])
    
    # Phase 2: Local search - try to improve by swapping out/in vectors
    # For each initial solution, try to replace selected vectors with unselected ones
    for sol_idx, sol in enumerate(initial_solutions):
        if time.time() > deadline - 15:
            break
            
        if len(sol) < best_count:
            continue
            
        selected = set(sol)
        unselected = set(range(n_vecs)) - selected
        selected_list = list(sol)
        
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            
            # Try to add unselected vectors
            unselected_list = list(unselected)
            rng = np.random.RandomState(sol_idx * 100 + 7)
            rng.shuffle(unselected_list)
            
            for uidx in unselected_list:
                if time.time() > deadline - 15:
                    break
                    
                v = vecs[uidx]
                sel_arr = np.array([vecs[i] for i in selected_list])
                dots = sel_arr @ v
                if np.any(dots > 2):
                    continue
                
                # Can add this vector
                selected_list.append(uidx)
                selected.add(uidx)
                unselected.remove(uidx)
                improved = True
                break
            
            if not improved:
                # Try swap: remove one selected, add one unselected
                # Try removing each selected vector and see if we can add more
                sel_list_copy = list(selected_list)
                rng.shuffle(sel_list_copy)
                
                for sidx in sel_list_copy:
                    if time.time() > deadline - 15:
                        break
                        
                    # Temporarily remove sidx
                    temp_selected = [i for i in selected_list if i != sidx]
                    temp_sel_set = set(temp_selected)
                    temp_unselected = set(range(n_vecs)) - temp_sel_set
                    temp_unselected_list = list(temp_unselected)
                    rng.shuffle(temp_unselected_list)
                    
                    best_add = None
                    best_add_idx = None
                    
                    # Try to add multiple vectors after removal
                    added_count = 0
                    new_selected = list(temp_selected)
                    new_sel_set = set(new_selected)
                    
                    for uidx in temp_unselected_list[:50]:  # Limit tries
                        if time.time() > deadline - 15:
                            break
                            
                        v = vecs[uidx]
                        if len(new_selected) > 0:
                            sel_arr = np.array([vecs[i] for i in new_selected])
                            dots = sel_arr @ v
                            if np.any(dots > 2):
                                continue
                        
                        new_selected.append(uidx)
                        new_sel_set.add(uidx)
                        added_count += 1
                    
                    # If we added more than 1 vector (net gain), accept
                    if added_count > 1:
                        selected_list = new_selected
                        selected = new_sel_set
                        unselected = set(range(n_vecs)) - selected
                        improved = True
                        break
            
            # Update best
            if len(selected_list) > best_count:
                best_count = len(selected_list)
                best_set = selected_list[:]
    
    # Convert best_set to array
    if best_count > 0:
        result = vecs[best_set].astype(np.int64)
    else:
        # Fallback: just return roots?
        roots = []
        for i in range(D):
            for j in range(i + 1, D):
                for si in [-1, 1]:
                    for sj in [-1, 1]:
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        roots.append(v)
        result = np.array(roots, dtype=np.int64)
        
    return result
