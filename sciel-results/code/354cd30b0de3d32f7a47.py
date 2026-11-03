import numpy as np
import itertools
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    D = 11
    
    # 1. Generate Candidate Pool: Weight-4 vectors
    # Norm^2 = 4. Max norm in set is 2.
    # Constraint: ||v_i - v_j||^2 >= 4 for all i != j.
    # For weight-4 vectors, this means dot(v_i, v_j) <= 2.
    
    candidates_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)
    
    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)
    
    # 2. Precompute Conflict Matrix
    c_int = candidates.astype(np.int32)
    dot = c_int @ c_int.T
    
    # Conflict if dot > 2
    conflict = (dot > 2).astype(np.uint8)
    np.fill_diagonal(conflict, 0)
    
    # Compatibility matrix H: 1 if compatible (dot <= 2), 0 otherwise.
    # H = 1 - conflict - I
    compatibility = 1 - conflict
    np.fill_diagonal(compatibility, 0)
    
    # 3. Spectral Analysis
    # Compute the leading eigenvector of the compatibility graph
    # We use eigsh on the sparse matrix for efficiency
    H_sparse = csr_matrix(compatibility)
    
    try:
        # Get the largest eigenvalue and its eigenvector
        # k=1 means we want the largest eigenvalue
        # sigma=0 means shift-invert mode, but for largest algebraic we can just use eigsh
        # However, eigsh finds eigenvalues near sigma. 
        # For largest eigenvalue, we can use eigsh(H_sparse, k=1, which='LA')
        vals, vecs = eigsh(H_sparse, k=1, which='LA')
        spectral_score = vecs[:, 0]
        # Ensure consistent sign (largest absolute value component is positive)
        if np.argmax(np.abs(spectral_score)):
            if spectral_score[np.argmax(np.abs(spectral_score))] < 0:
                spectral_score = -spectral_score
    except Exception:
        # Fallback to degree centrality if eigsh fails
        spectral_score = compatibility.sum(axis=1).astype(float)
        
    # Normalize spectral score to [0, 1]
    s_min = spectral_score.min()
    s_max = spectral_score.max()
    if s_max > s_min:
        spectral_score_norm = (spectral_score - s_min) / (s_max - s_min)
    else:
        spectral_score_norm = np.zeros(n)

    # Precompute conflict neighbor lists for fast updates
    conflict_neighbors = [np.where(conflict[i] == 1)[0] for i in range(n)]
    
    best_count = 0
    best_selected = np.zeros(n, dtype=np.uint8)
    
    rng = np.random.default_rng(42)
    
    def greedy_construct(temperature=0.0):
        """Greedy construction with spectral bias and optional randomness."""
        selected = np.zeros(n, dtype=np.uint8)
        remaining = np.ones(n, dtype=np.uint8)
        
        count = 0
        while True:
            if time.time() > deadline - 10.0:
                break
                
            rem_idx = np.where(remaining > 0)[0]
            if len(rem_idx) == 0:
                break
                
            # Calculate effective degree in the remaining subgraph
            deg = (conflict * remaining[np.newaxis, :]).sum(axis=1)
            deg[remaining == 0] = n + 1
            
            min_deg = deg.min()
            if min_deg >= n:
                break
                
            if temperature > 0:
                # Soft selection: combine spectral score and degree
                # We prefer low degree and high spectral score
                active_mask = remaining > 0
                if not active_mask.any():
                    break
                    
                # Score = w1 * (1 - deg_norm) + w2 * spectral_score_norm
                # Let's define a combined score
                deg_active = deg[active_mask]
                spec_active = spectral_score_norm[active_mask]
                
                # Normalize deg to [0,1] within active set
                d_min = deg_active.min()
                d_max = deg_active.max()
                if d_max > d_min:
                    deg_norm = (deg_active - d_min) / (d_max - d_min)
                else:
                    deg_norm = np.zeros_like(deg_active)
                
                # We want low deg, so use 1 - deg_norm
                # Weight spectral score heavily?
                # Let's try: Score = (1 - deg_norm) + 0.5 * spec_active
                scores = (1 - deg_norm) + 0.5 * spec_active
                
                # Add noise based on temperature
                noise = rng.exponential(scale=temperature, size=len(scores))
                scores += noise
                
                # Pick max score
                best_local_idx = np.argmax(scores)
                pick_idx = rem_idx[best_local_idx]
            else:
                # Deterministic: Pick min degree, break ties with spectral score
                candidates_to_pick = np.where((deg == min_deg) & (remaining > 0))[0]
                if len(candidates_to_pick) == 0:
                    break
                # Among min-degree candidates, pick highest spectral score
                spec_vals = spectral_score_norm[candidates_to_pick]
                best_local_idx = np.argmax(spec_vals)
                pick_idx = candidates_to_pick[best_local_idx]
            
            selected[pick_idx] = 1
            remaining[pick_idx] = 0
            remaining[conflict[pick_idx] == 1] = 0
            count += 1
            
        return selected

    def local_search(selected):
        """Local search with incremental incompat tracking."""
        sel_indices = np.where(selected == 1)[0]
        if len(sel_indices) == 0:
            return selected, 0
            
        # Initialize incompat: for each node, count conflicts with selected set
        incompat = np.zeros(n, dtype=np.int32)
        for v in sel_indices:
            incompat[conflict_neighbors[v]] += 1
        
        improved = True
        max_moves = 800
        move_count = 0
        moves_made = 0
        
        while improved and move_count < max_moves and time.time() < deadline - 5.0:
            improved = False
            move_count += 1
            
            sel_indices = np.where(selected == 1)[0]
            k_sel = len(sel_indices)
            
            if k_sel == 0:
                break
            
            # Try 1-for-2 moves
            order = rng.permutation(k_sel)
            
            for idx in order:
                if time.time() > deadline - 5.0:
                    break
                    
                v_idx_in_S = idx
                v = sel_indices[v_idx_in_S]
                
                # Candidates u such that u is compatible with S \ {v}
                cond_0 = (incompat == 0)
                cond_1 = (incompat == 1) & (conflict[:, v] == 1)
                Cand_v = (cond_0 | cond_1) & (selected == 0)
                
                cand_indices = np.where(Cand_v > 0)[0]
                
                if len(cand_indices) < 2:
                    continue
                
                found_pair = False
                u_val = -1
                w_val = -1
                
                # Fast pair finding
                if len(cand_indices) <= 200:
                    for i_u in range(len(cand_indices)):
                        u = cand_indices[i_u]
                        conflicts_with_cands = conflict[u, cand_indices]
                        compat_positions = np.where(conflicts_with_cands == 0)[0]
                        compat_positions = compat_positions[compat_positions != i_u]
                        if len(compat_positions) > 0:
                            w = cand_indices[compat_positions[0]]
                            u_val = u
                            w_val = w
                            found_pair = True
                            break
                else:
                    # Large candidate set: use random sampling
                    num_samples = min(500, len(cand_indices) * 2)
                    for _ in range(num_samples):
                        i_u = rng.integers(len(cand_indices))
                        i_w = rng.integers(len(cand_indices))
                        if i_u == i_w:
                            continue
                        u = cand_indices[i_u]
                        w = cand_indices[i_w]
                        if conflict[u, w] == 0:
                            u_val = u
                            w_val = w
                            found_pair = True
                            break
                    
                    if not found_pair:
                        check_range = min(100, len(cand_indices))
                        for i_u in range(check_range):
                            u = cand_indices[i_u]
                            conflicts_with_cands = conflict[u, cand_indices]
                            compat_positions = np.where(conflicts_with_cands == 0)[0]
                            compat_positions = compat_positions[compat_positions != i_u]
                            if len(compat_positions) > 0:
                                w = cand_indices[compat_positions[0]]
                                u_val = u
                                w_val = w
                                found_pair = True
                                break
                
                if found_pair:
                    u = u_val
                    w = w_val
                    
                    # Perform move: remove v, add u, w
                    incompat[conflict_neighbors[v]] -= 1
                    incompat[conflict_neighbors[u]] += 1
                    incompat[conflict_neighbors[w]] += 1
                    
                    selected[v] = 0
                    selected[u] = 1
                    selected[w] = 1
                    improved = True
                    moves_made += 1
                    break  # Restart while loop with new S
            
            # If no 1-for-2 found, try 1-for-1 swaps
            if not improved:
                if k_sel > 0:
                    for _ in range(5):
                        v_idx_in_S = rng.integers(k_sel)
                        v = sel_indices[v_idx_in_S]
                        
                        cond_0 = (incompat == 0)
                        cond_1 = (incompat == 1) & (conflict[:, v] == 1)
                        Cand_v = (cond_0 | cond_1) & (selected == 0)
                        
                        cand_indices = np.where(Cand_v > 0)[0]
                        
                        if len(cand_indices) > 0:
                            # Pick candidate with highest spectral score
                            spec_vals = spectral_score_norm[cand_indices]
                            best_local_idx = np.argmax(spec_vals)
                            u = cand_indices[best_local_idx]
                            
                            incompat[conflict_neighbors[v]] -= 1
                            incompat[conflict_neighbors[u]] += 1
                            
                            selected[v] = 0
                            selected[u] = 1
                            improved = True
                            moves_made += 1
                            break

        return selected, moves_made

    # Multi-start search
    num_restarts = 1000
    
    for restart in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Vary temperature
        if restart < 50:
            temp = 0.0
        elif restart < 200:
            temp = 0.5
        elif restart < 500:
            temp = 1.0
        else:
            temp = 2.0
        
        # Phase 1: Greedy
        selected = greedy_construct(temperature=temp)
        
        # Phase 2: Local Search
        selected, moves = local_search(selected)
        
        current_count = selected.sum()
        if current_count > best_count:
            best_count = current_count
            best_selected = selected.copy()
            
            # Extra refinement
            if current_count > best_count - 5 and time.time() < deadline - 30.0:
                selected2, _ = local_search(best_selected.copy())
                c2 = selected2.sum()
                if c2 > best_count:
                    best_count = c2
                    best_selected = selected2.copy()

    # 4. Extract Result
    sel_indices = np.where(best_selected == 1)[0]
    if len(sel_indices) == 0:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)
        
    result = candidates[sel_indices.astype(np.int64)]
    return result.astype(np.int64)

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} vectors")
