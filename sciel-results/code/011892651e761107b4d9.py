import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Generate candidates with squared norm 4
    candidates = []
    
    # Type 1: One 2, rest 0 (squared norm 4)
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four 1s, rest 0 (squared norm 4)
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # Conflict matrix: 1 if dot > 2 (and i != j)
    # Note: M[i,i] = 4, which is > 2, but we don't conflict with ourselves.
    Conflict = (M > 2).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)
    
    # NonConflict matrix for faster pair finding
    # NonConflict[i, j] = 1 if i != j and not Conflict[i, j]
    NonConflict = (Conflict == 0).astype(np.uint8)
    np.fill_diagonal(NonConflict, 0)
    
    best_set = []
    best_size = 0
    
    def run_greedy(seed):
        if time.time() > deadline:
            return []
            
        rng = np.random.default_rng(seed)
        active = np.ones(N, dtype=bool)
        deg = Conflict.sum(axis=1).astype(np.int64)
        selected = []
        
        while active.any():
            if time.time() > deadline:
                break
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
                
            active_deg = deg[active_indices]
            min_deg = np.min(active_deg)
            candidates_for_pick = active_indices[active_deg == min_deg]
            idx = rng.choice(candidates_for_pick)
            
            selected.append(idx)
            
            # Remove idx and its neighbors
            neighbors_mask = Conflict[idx].astype(bool)
            removed_mask = np.zeros(N, dtype=bool)
            removed_mask[idx] = True
            removed_mask[neighbors_mask & active] = True
            
            removed_indices = np.where(removed_mask)[0]
            
            # Update degrees
            if len(removed_indices) > 0:
                removal_counts = Conflict[removed_indices, :].sum(axis=0).astype(np.int64)
                deg[active] -= removal_counts[active]
                
            active[removed_mask] = False
            
        return selected

    def local_search(selected_indices, max_iters=50):
        if time.time() > deadline:
            return selected_indices
            
        selected_set = set(selected_indices)
        S_mask = np.zeros(N, dtype=bool)
        for i in selected_set:
            S_mask[i] = True
            
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            indices_list = list(selected_set)
            rng_ls = np.random.default_rng(iters + 1000)
            rng_ls.shuffle(indices_list)
            
            for i in indices_list:
                if time.time() > deadline:
                    return list(selected_set)
                
                # Calculate compatible nodes for removing i
                # S_without_i
                S_mask[i] = False
                # conflicts_with_S_minus_i: True if j conflicts with any node in S \ {i}
                # Conflict[:, S_mask] has shape (N, |S|)
                # We want any() along axis 1
                if np.any(S_mask):
                    conflicts_with_rest = Conflict[:, S_mask].any(axis=1)
                else:
                    conflicts_with_rest = np.zeros(N, dtype=bool)
                
                # Compatible: not in S, and not conflicting with S \ {i}
                compatible = ~conflicts_with_rest & ~S_mask
                
                # Restore S_mask[i] for next iteration logic or just keep it modified?
                # We are checking swap, so we assume i is removed.
                # If we find a swap, we will update S_mask.
                
                C_indices = np.where(compatible)[0]
                if len(C_indices) < 2:
                    S_mask[i] = True # Restore if no swap
                    continue
                
                # Find a pair j, k in C_indices such that NonConflict[j, k] == 1
                # Strategy: Pick random j from C_indices, check if any k works.
                found_pair = False
                # Try a few random picks to speed up
                for _ in range(10):
                    if time.time() > deadline:
                        return list(selected_set)
                    j_idx = rng_ls.choice(len(C_indices))
                    j = C_indices[j_idx]
                    # Check NonConflict[j, C_indices]
                    valid_ks = np.where(NonConflict[j, C_indices] == 1)[0]
                    # Exclude j itself (though NonConflict diagonal is 0, so j won't be there)
                    if len(valid_ks) > 0:
                        k_idx = rng_ls.choice(valid_ks)
                        k = C_indices[k_idx]
                        
                        # Perform Swap
                        selected_set.remove(i)
                        selected_set.add(j)
                        selected_set.add(k)
                        
                        # Update S_mask
                        S_mask[i] = False
                        S_mask[j] = True
                        S_mask[k] = True
                        
                        improved = True
                        found_pair = True
                        break
                    # If no k found for this j, try another j
                    
                if not found_pair:
                    # Restore S_mask[i] if we didn't swap
                    S_mask[i] = True
                    continue
                    
                if found_pair:
                    break # Restart local search loop with new set
                
        return list(selected_set)

    def spectral_seed():
        """
        Use spectral clustering on the compatibility graph to find a dense cluster
        that can serve as a high-quality initial seed for local search.
        """
        if time.time() > deadline:
            return []
            
        # The compatibility graph is defined by NonConflict (1 if compatible, 0 if conflict)
        # We want to find a large independent set in the Conflict graph,
        # which is equivalent to a large clique in the NonConflict graph.
        
        # Use the leading eigenvector of the Laplacian of the NonConflict graph?
        # Or better: Use the adjacency matrix of NonConflict directly.
        # A dense cluster in the NonConflict graph corresponds to a region with high degree.
        
        # Let's try a simple spectral method:
        # Compute the Fiedler vector (2nd smallest eigenvalue) of the Laplacian of NonConflict?
        # Actually, for finding a dense subgraph, we can look at the principal eigenvector of the adjacency matrix.
        
        A = NonConflict.astype(np.float64)
        
        # Compute the leading eigenvector of A
        # scipy is available
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import csr_matrix
        
        # A is dense, but let's use dense eig for simplicity if N is small enough
        # N is around 22 + 330*16 = 22 + 5280 = 5302
        # Dense eig on 5302x5302 might be slow (O(N^3)).
        # Let's use scipy.sparse.linalg.eigsh for the largest eigenvalue.
        
        try:
            # We want the eigenvector corresponding to the largest eigenvalue of A
            # This often highlights the largest connected component or densest cluster.
            # However, eigsh requires a symmetric matrix and works on sparse matrices efficiently.
            A_sparse = csr_matrix(A)
            # k=1 for the largest eigenvalue
            # sigma=0 for shift-invert to find largest magnitude? No, just 'LA' for largest algebraic.
            # But A is non-negative, so largest algebraic is the spectral radius.
            vals, vecs = eigsh(A_sparse, k=1, which='LA')
            # The eigenvector is in vecs[:, 0]
            # Nodes with high positive values in the eigenvector are likely in the dense cluster.
            # Let's threshold the eigenvector.
            
            v = vecs[:, 0]
            # Normalize v to [0, 1] for easier thresholding? Or just use top-k?
            # Let's pick the top 500 nodes by eigenvector value.
            # Or pick nodes with value > mean.
            
            # Let's try picking the top 1000 nodes.
            top_k = min(1000, N)
            top_indices = np.argsort(v)[-top_k:]
            
            # Now, we need to extract the largest independent set (clique in NonConflict) from this subgraph.
            # We can run a greedy algorithm on this subset.
            
            # Create a mask for the subset
            subset_mask = np.zeros(N, dtype=bool)
            subset_mask[top_indices] = True
            
            # Run greedy on the subset only
            # We can reuse run_greedy logic but restricted to subset?
            # Or just run a simple greedy on the induced subgraph.
            
            # Let's implement a quick greedy for the subset
            active = subset_mask.copy()
            deg = Conflict[subset_mask].sum(axis=1).astype(np.int64) # Degree within subset? No, Conflict is global.
            # We need degree within the subset.
            # Conflict_subset = Conflict[np.ix_(top_indices, top_indices)]
            # deg_subset = Conflict_subset.sum(axis=1)
            
            # Actually, let's just run the standard greedy but initialize with a bias towards the spectral cluster?
            # Or simpler: Run greedy on the full graph but start with the spectral cluster nodes?
            
            # Let's try: Start with the top 500 nodes from spectral, then try to add more?
            # This might not work well if they conflict.
            
            # Better: Run a greedy algorithm restricted to the top 2000 nodes identified by spectral.
            
            # Let's refine: Pick top 2000 nodes.
            top_k = min(2000, N)
            top_indices = np.argsort(v)[-top_k:]
            
            # Run a greedy independent set on this induced subgraph
            # We need the Conflict matrix for this subset
            idxs = top_indices
            C_sub = Conflict[np.ix_(idxs, idxs)]
            N_sub = len(idxs)
            
            active = np.ones(N_sub, dtype=bool)
            deg = C_sub.sum(axis=1).astype(np.int64)
            selected_local = []
            
            while active.any():
                if time.time() > deadline:
                    break
                active_indices = np.where(active)[0]
                if len(active_indices) == 0:
                    break
                    
                active_deg = deg[active_indices]
                min_deg = np.min(active_deg)
                candidates_for_pick = active_indices[active_deg == min_deg]
                # Pick the one with smallest degree, tie-break by original index for determinism?
                # Or random? Let's use random for variety.
                idx_local = candidates_for_pick[np.random.randint(len(candidates_for_pick))]
                
                selected_local.append(idxs[idx_local])
                
                # Remove idx_local and its neighbors in the subgraph
                neighbors_mask = C_sub[idx_local].astype(bool)
                removed_mask = np.zeros(N_sub, dtype=bool)
                removed_mask[idx_local] = True
                removed_mask[neighbors_mask & active] = True
                
                removed_indices = np.where(removed_mask)[0]
                
                if len(removed_indices) > 0:
                    removal_counts = C_sub[removed_indices, :].sum(axis=0).astype(np.int64)
                    deg[active] -= removal_counts[active]
                    
                active[removed_mask] = False
                
            return selected_local
            
        except Exception as e:
            # Fallback if spectral fails
            return []

    # Run spectral seed first
    spectral_res = spectral_seed()
    if spectral_res:
        spectral_res_improved = local_search(spectral_res)
        if len(spectral_res_improved) > best_size:
            best_size = len(spectral_res_improved)
            best_set = spectral_res_improved

    # Run greedy with multiple seeds
    for s in range(300):
        if time.time() > deadline:
            break
        
        res = run_greedy(s)
        if not res:
            continue
            
        # Apply local search
        res_improved = local_search(res)
        
        if len(res_improved) > best_size:
            best_size = len(res_improved)
            best_set = res_improved
            
    if not best_set:
        best_set = list(range(22))
        
    return candidates[best_set].astype(np.int64)
