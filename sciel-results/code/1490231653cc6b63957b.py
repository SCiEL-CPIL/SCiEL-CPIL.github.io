<<<<<<< SEARCH
    # Multiple restarts with varied strategies
    n_restarts = 0
    max_restarts = 300
    
    def greedy_max_degree_supports():
        """Greedy independent set with max-degree heuristic (opposite of min)."""
        in_set = np.zeros(n_supports, dtype=bool)
        alive = np.ones(n_supports, dtype=bool)
        degrees = neighbor_counts_sup.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            max_deg = np.max(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == max_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors_sup[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= support_conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set
    
    def random_init_supports(density=0.15):
        """Random initialization with given density, then repair conflicts."""
        in_set = np.zeros(n_supports, dtype=bool)
        candidates = rng.integers(0, n_supports, size=int(n_supports * density * 3))
        
        for s in candidates:
            if not in_set[s] and not np.any(support_conflicts[s] & in_set):
                in_set[s] = True
        
        return in_set
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        
        # Choose initialization strategy
        strategy = n_restarts % 3
        if strategy == 0:
            in_set = greedy_min_degree_supports()
        elif strategy == 1:
            in_set = greedy_max_degree_supports()
        else:
            in_set = random_init_supports()
            
        current_size = in_set.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_support_set = in_set.copy()
        
        # SA Local Search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        sa_time = min(remaining_time * 0.5, 12.0)
        if sa_time < 2:
            sa_time = 2.0
            
        sa_result, sa_size = sa_local_search_supports(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_support_set = sa_result.copy()
        
        # Final greedy polish: try to add any remaining non-conflicting supports
        if time.time() < deadline - 3:
            polish_set = best_support_set.copy()
            polished = True
            while polished and time.time() < deadline - 3:
                polished = False
                not_in = np.where(~polish_set)[0]
                if len(not_in) == 0:
                    break
                # Find supports with zero conflicts with current set
                for s in not_in:
                    if not np.any(support_conflicts[s] & polish_set):
                        polish_set[s] = True
                        polished = True
                        if polish_set.sum() > best_size:
                            best_size = polish_set.sum()
                            best_support_set = polish_set.copy()
=======
    # Multiple restarts with varied strategies
    n_restarts = 0
    max_restarts = 300
    
    # Spectral embedding for better initialization
    # Compute eigenvectors of the conflict graph's Laplacian
    L = np.eye(n_supports) - support_conflicts.astype(float)
    # Normalize: D^{-1/2} L D^{-1/2}
    degs = L.sum(axis=1)
    degs_safe = np.where(degs > 0, degs, 1.0)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degs_safe))
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt
    
    # Get top eigenvectors (smallest eigenvalues give community structure)
    try:
        eigvals, eigvecs = np.linalg.eigh(L_sym)
        # Use the eigenvectors corresponding to the smallest non-zero eigenvalues
        # Skip the first (constant) eigenvector
        spectral_embed = eigvecs[:, 1:8]  # 7-dimensional embedding
    except:
        spectral_embed = None
    
    def spectral_cluster_init(n_clusters=6):
        """Use spectral embedding + k-means to find independent set seeds."""
        if spectral_embed is None or n_restarts >= max_restarts:
            return random_init_supports()
        
        # Simple k-means on spectral embedding
        X = spectral_embed
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
        
        # Initialize centroids randomly
        centroid_indices = rng.choice(n_supports, size=n_clusters, replace=False)
        centroids = X_norm[centroid_indices].copy()
        
        labels = np.zeros(n_supports, dtype=int)
        for _ in range(20):
            # Assign to nearest centroid
            dists = np.linalg.norm(X_norm[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
            new_labels = np.argmin(dists, axis=1)
            
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            
            # Update centroids
            for c in range(n_clusters):
                mask = labels == c
                if mask.sum() > 0:
                    centroids[c] = X_norm[mask].mean(axis=0)
        
        # For each cluster, greedily build an independent set
        best_cluster_set = np.zeros(n_supports, dtype=bool)
        best_cluster_size = 0
        
        for c in range(n_clusters):
            cluster_members = np.where(labels == c)[0]
            if len(cluster_members) == 0:
                continue
            
            # Greedy within cluster: pick nodes with fewest conflicts to others in cluster
            in_set = np.zeros(n_supports, dtype=bool)
            remaining = list(cluster_members)
            
            while remaining and time.time() < deadline - 5:
                # Score each remaining node by number of conflicts with already-selected
                scores = []
                for node in remaining:
                    conflicts = np.sum(support_conflicts[node] & in_set)
                    scores.append(conflicts)
                
                min_conflict = min(scores)
                candidates = [remaining[i] for i in range(len(remaining)) if scores[i] == min_conflict]
                chosen = candidates[rng.integers(len(candidates))]
                
                in_set[chosen] = True
                remaining = [n for n in remaining if not support_conflicts[n, chosen]]
            
            if in_set.sum() > best_cluster_size:
                best_cluster_size = in_set.sum()
                best_cluster_set = in_set.copy()
        
        return best_cluster_set
    
    def greedy_max_degree_supports():
        """Greedy independent set with max-degree heuristic (opposite of min)."""
        in_set = np.zeros(n_supports, dtype=bool)
        alive = np.ones(n_supports, dtype=bool)
        degrees = neighbor_counts_sup.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            max_deg = np.max(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == max_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors_sup[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= support_conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set
    
    def random_init_supports(density=0.15):
        """Random initialization with given density, then repair conflicts."""
        in_set = np.zeros(n_supports, dtype=bool)
        candidates = rng.integers(0, n_supports, size=int(n_supports * density * 3))
        
        for s in candidates:
            if not in_set[s] and not np.any(support_conflicts[s] & in_set):
                in_set[s] = True
        
        return in_set
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        
        # Choose initialization strategy
        strategy = n_restarts % 4
        if strategy == 0:
            in_set = greedy_min_degree_supports()
        elif strategy == 1:
            in_set = greedy_max_degree_supports()
        elif strategy == 2:
            in_set = random_init_supports()
        else:
            in_set = spectral_cluster_init()
            
        current_size = in_set.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_support_set = in_set.copy()
        
        # SA Local Search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        sa_time = min(remaining_time * 0.5, 12.0)
        if sa_time < 2:
            sa_time = 2.0
            
        sa_result, sa_size = sa_local_search_supports(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_support_set = sa_result.copy()
        
        # Final greedy polish: try to add any remaining non-conflicting supports
        if time.time() < deadline - 3:
            polish_set = best_support_set.copy()
            polished = True
            while polished and time.time() < deadline - 3:
                polished = False
                not_in = np.where(~polish_set)[0]
                if len(not_in) == 0:
                    break
                # Find supports with zero conflicts with current set
                for s in not_in:
                    if not np.any(support_conflicts[s] & polish_set):
                        polish_set[s] = True
                        polished = True
                        if polish_set.sum() > best_size:
                            best_size = polish_set.sum()
                            best_support_set = polish_set.copy()
>>>>>>> REPLACE

Now let me also improve the local search with better perturbation strategies:

