"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def construct():
    start_time = time.time()
    deadline = start_time + 112  # Safety margin before 150s kill

    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    # Generate W4 supports (weight 4, entries in {-1,0,1}, norm^2=4)
    # We group by support. For each support, we can include all 16 sign variants
    # if the supports themselves are pairwise compatible (intersection <= 2).
    
    supports = list(itertools.combinations(range(D), 4))
    n_supports = len(supports)
    
    # Map support to index
    support_to_idx = {s: i for i, s in enumerate(supports)}
    
    # Build conflict graph on supports
    # Two supports conflict if they share >= 3 positions
    conflict_matrix = np.zeros((n_supports, n_supports), dtype=bool)
    
    for i in range(n_supports):
        si = set(supports[i])
        for j in range(i + 1, n_supports):
            sj = set(supports[j])
            if len(si & sj) >= 3:
                conflict_matrix[i, j] = True
                conflict_matrix[j, i] = True
    
    # Precompute neighbors for faster SA
    neighbors = [np.where(conflict_matrix[i])[0] for i in range(n_supports)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(n_supports, dtype=bool)
    
    def greedy_min_degree():
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(n_supports, dtype=bool)
        alive = np.ones(n_supports, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflict_matrix[rem][:, to_remove].sum(axis=1)
                
        return in_set
    
    def spectral_embedding_init():
        """Spectral embedding of conflict graph, then spherical k-means for independent set init."""
        if time.time() > deadline - 10:
            return greedy_min_degree()
        
        # Build sparse adjacency matrix
        adj = csr_matrix(conflict_matrix.astype(np.float64))
        
        # Compute Fiedler vector (second eigenvector) for spectral embedding
        try:
            # Use eigsh to find the second smallest eigenvalue/eigenvector
            # For a connected graph, the Fiedler vector gives a 1D embedding
            # We'll use a few eigenvectors for better separation
            k_eig = min(8, n_supports - 1)
            # sigma=0 finds smallest magnitude eigenvalues
            # We want the eigenvectors corresponding to small eigenvalues
            # (Laplacian eigenvectors for clustering)
            
            # Compute Laplacian L = D - A
            degree_vec = np.array(conflict_matrix.sum(axis=1)).flatten()
            laplacian = csr_matrix(np.diag(degree_vec)) - adj
            
            # Get k smallest eigenvalues/eigenvectors
            eigvals, eigvecs = eigsh(laplacian, k=k_eig, sigma=0, which='LM', 
                                      return_eigenvectors=True)
            
            # Use eigenvectors (excluding the trivial constant one) for embedding
            # The first eigenvector is typically constant (for connected graph)
            embedding = eigvecs[:, 1:]  # Skip the trivial eigenvector
            
            # Normalize rows for spherical k-means
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embedding = embedding / norms
            
        except Exception:
            # Fallback to random embedding if spectral fails
            embedding = rng.standard_normal((n_supports, 4))
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embedding = embedding / norms
        
        # Spherical k-means to partition into clusters
        # We want to find a large independent set, so we try to cluster
        # and pick the largest cluster that is (approximately) independent
        
        n_clusters = 4
        max_iter = 50
        
        # Initialize centroids randomly from data points
        idx = rng.choice(n_supports, size=n_clusters, replace=False)
        centroids = embedding[idx].copy()
        
        for it in range(max_iter):
            # Assign points to nearest centroid (spherical: cosine similarity)
            sims = embedding @ centroids.T
            labels = np.argmax(sims, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for c in range(n_clusters):
                mask = labels == c
                if np.sum(mask) > 0:
                    new_centroids[c] = embedding[mask].sum(axis=0)
                    norm = np.linalg.norm(new_centroids[c])
                    if norm > 0:
                        new_centroids[c] /= norm
                    else:
                        new_centroids[c] = centroids[c]
                else:
                    new_centroids[c] = centroids[c]
            
            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids
        
        # Find the largest cluster that has the fewest internal conflicts
        # Try each cluster, greedily remove conflicting nodes
        best_cluster_set = np.zeros(n_supports, dtype=bool)
        best_cluster_size = 0
        
        for c in range(n_clusters):
            mask = labels == c
            cluster_nodes = np.where(mask)[0]
            if len(cluster_nodes) == 0:
                continue
            
            # Greedily select independent set from this cluster
            cluster_in_set = np.zeros(len(cluster_nodes), dtype=bool)
            remaining = np.ones(len(cluster_nodes), dtype=bool)
            
            for _ in range(len(cluster_nodes)):
                if not np.any(remaining):
                    break
                
                # Find node with min degree within remaining
                rem_indices = cluster_nodes[remaining]
                min_deg = np.inf
                best_idx = -1
                for i, node in enumerate(rem_indices):
                    deg = np.sum(conflict_matrix[node] & mask & remaining)
                    if deg < min_deg:
                        min_deg = deg
                        best_idx = i
                
                if best_idx == -1:
                    break
                
                chosen = rem_indices[best_idx]
                cluster_in_set[best_idx] = True
                # Remove chosen and its neighbors from remaining
                to_remove = np.where(conflict_matrix[chosen] & mask)[0]
                for tr in to_remove:
                    if tr in cluster_nodes:
                        idx_in_cluster = np.where(cluster_nodes == tr)[0]
                        if len(idx_in_cluster) > 0:
                            remaining[idx_in_cluster[0]] = False
                remaining[best_idx] = False
            
            if cluster_in_set.sum() > best_cluster_size:
                best_cluster_size = cluster_in_set.sum()
                best_cluster_set = np.zeros(n_supports, dtype=bool)
                best_cluster_set[cluster_nodes[cluster_in_set]] = True
        
        return best_cluster_set
    
    def sa_local_search(in_set_init, max_time=10.0):
        """Min-conflict simulated annealing with 2-for-1 swap moves."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        
        T = 3.0
        cooling = 0.9995
        min_T = 0.05
        
        # count_conflicts[s] = number of neighbors of s that are in in_set
        count_conflicts = np.zeros(n_supports, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        while time.time() < sa_deadline:
            # Choose move type: 70% insert, 30% 2-for-1 swap
            if rng.random() < 0.7:
                # Single insert move
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    break
                    
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                
                min_cands = not_in_set[counts == min_count]
                s = min_cands[rng.integers(len(min_cands))]
                
                k = min_count
                delta = 1 - k
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.5
                else:
                    accept = rng.random() < np.exp(delta / T)
                    
                if accept:
                    to_remove = np.where(conflict_matrix[s] & in_set)[0]
                    
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                    
                    in_set[s] = True
                    current_size += delta
                    count_conflicts[neighbors[s]] += 1
                    
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1
            else:
                # 2-for-1 swap: remove 2 conflicting nodes, add 1 new node
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    break
                
                # Sample a few candidates for the new node
                n_sample = min(5, len(not_in_set))
                sample_idx = rng.choice(len(not_in_set), size=n_sample, replace=False)
                sample_nodes = not_in_set[sample_idx]
                
                swap_accepted = False
                for s in sample_nodes:
                    # Get conflicts of s that are currently in the set
                    conflicts_in_set = np.where(conflict_matrix[s] & in_set)[0]
                    k = len(conflicts_in_set)
                    
                    if k < 2:
                        continue  # Need at least 2 to remove for a 2-for-1 swap
                    
                    # Check if removing these k conflicts and adding s improves size
                    delta = 1 - k
                    
                    # Accept if delta >= 0 or with probability exp(delta/T)
                    if delta >= 0:
                        accept = True
                    else:
                        accept = rng.random() < np.exp(delta / T)
                    
                    if accept:
                        # Perform the swap
                        to_remove = conflicts_in_set
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                        
                        in_set[s] = True
                        current_size += delta
                        count_conflicts[neighbors[s]] += 1
                        
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()
                            stuck_counter = 0
                        else:
                            stuck_counter += 1
                        
                        swap_accepted = True
                        break
                
                if not swap_accepted:
                    stuck_counter += 1
            
            if stuck_counter > 1500:
                T = max(T, 2.0)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    # Multiple restarts
    n_restarts = 0
    max_restarts = 200
    
    # Use spectral embedding for the first few restarts
    use_spectral = True
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Initialization: spectral for first 5 restarts, then greedy
        if use_spectral and n_restarts <= 5:
            in_set = spectral_embedding_init()
        else:
            in_set = greedy_min_degree()
        
        current_size = in_set.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 2. SA Local Search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        sa_time = min(remaining_time * 0.6, 15.0)
        if sa_time < 2:
            sa_time = 2.0
            
        sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        
        # 3. Second SA pass on best found so far (polish)
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 5 and sa_size > current_size:
            polish_time = min(remaining_time * 0.5, 10.0)
            if polish_time > 3:
                polish_result, polish_size = sa_local_search(sa_result, max_time=polish_time)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()
    
    # Build final vector list from selected supports
    selected_support_indices = np.where(best_in_set)[0]
    final_vectors = []
    
    for si in selected_support_indices:
        support = supports[si]
        # Generate all 16 sign variants for this support
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(support, signs):
                v[pos] = s
            final_vectors.append(v)
    
    # Add all W1 vectors (22 of them)
    final_vectors.extend(v1_candidates)
    
    result = np.array(final_vectors, dtype=np.int64)
    
    # Fallback: if something went wrong, use deterministic construction
    if len(result) == 0:
        # Deterministic fallback: pick non-conflicting supports greedily
        selected_supports = []
        for s in supports:
            valid = True
            for t in selected_supports:
                intersection = set(s).intersection(t)
                if len(intersection) > 2:
                    valid = False
                    break
            if valid:
                selected_supports.append(s)
        
        fallback_vectors = []
        for s in selected_supports:
            for signs in itertools.product([1, -1], repeat=4):
                v = [0] * D
                for pos, sign in zip(s, signs):
                    v[pos] = sign
                fallback_vectors.append(v)
        
        # Add W1 vectors
        for pos in range(D):
            for sign in [1, -1]:
                v = [0] * D
                v[pos] = 2 * sign
                fallback_vectors.append(v)
        
        result = np.array(fallback_vectors, dtype=np.int64)
    
    return result

# EVOLVE-BLOCK-END
